"""Fixed, loopback-only Qwen3-VL chat adapter with offline image understanding."""
from __future__ import annotations
import base64
import gc
import functools
import hashlib
from io import BytesIO
import json
import os
import re
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid
import wave
from typing import Annotated

ROOT=Path(os.environ['VF_RUNTIME_ROOT']).resolve()
MODEL=Path(os.environ['VF_QWEN_MODEL']).resolve()
MODEL_ID='Qwen/Qwen3-VL-2B-Instruct'
PROFILE='screenplay-qwen2b'
SERVICE_ID='video-factory-screenplay-qwen-v1'
# ``None`` means the checkpoint's own advertised context window.  Wrappers may
# narrow this only when a smaller value is an explicit physical model limit.
MAX_CONTEXT_TOKENS=None
MAX_PROMPT_TOKENS=None
# Text is admitted according to the model's real token context after the
# chat template is rendered. A character-count ceiling is neither a reliable
# token estimate nor an appropriate workflow policy.
# Image sizing is delegated to the model's saved processor configuration and
# the actual combined prompt/context calculation.  There is no service-side
# edge, count, byte, or character policy ceiling.
VRAM_PREFLIGHT=False
SAFE_TEXT_SDPA=False
# Enabled only by the fixed H3-IR wrapper, never by the screenplay/JSON paths.
H3IR_PROSE_REPEAT_GUARD=False
PROSE_OUTPUT_REPETITION_PENALTY=1.1
PROSE_REPEAT_MIN_PERIOD=32
PROSE_REPEAT_MAX_PERIOD=256
PROSE_REPEAT_OCCURRENCES=3
PROSE_REPEAT_CHECK_EVERY=8
GPU_LEASE_PATH=ROOT/'cache/qwen/gpu-inference.lock'
SCRIPT_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOURCE_COMPONENTS={'base_script_sha256':SCRIPT_SHA}
os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
 'HF_HOME':str(ROOT/'cache/huggingface'),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
 'TORCH_HOME':str(ROOT/'cache/torch'),'CUDA_CACHE_PATH':str(ROOT/'cache/cuda'),
 'TEMP':str(ROOT/'tmp'),'TMP':str(ROOT/'tmp')})
sys.dont_write_bytecode=True
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Pillow's default decompression-bomb threshold is a static pixel policy.  The
# adapter instead reads headers without decoding, then applies the dynamic
# host/cgroup-memory preflight below before the model processor can allocate
# image buffers.
Image.MAX_IMAGE_PIXELS=None

app=FastAPI(title='VideoFactory fixed local screenplay model')
lock=threading.Lock()
# Decode one request's inline image payloads at a time.  This is separate from
# the GPU lease: it prevents two HTTP workers from allocating large decoded
# image buffers simultaneously before either can reach the inference lock.
input_decode_lock=threading.Lock()
weight_verified=False
weights_receipt=None
grammar_tokenizer_data=None
grammar_control_fragments={}
loaded=False
inference_phase='idle'
gpu_release_status='not_used'
retained_gpu_lease=None
last_gpu_cleanup=None

class ChatTemplateOptions(BaseModel):
    model_config=ConfigDict(extra='forbid')
    enable_thinking:bool=Field(default=False,strict=True)

StopText=Annotated[str,Field(strict=True,min_length=1)]

class ChatRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model:str
    trace_id:str|None=None
    messages:list[dict]=Field(min_length=1)
    # ``None`` asks the adapter to use all remaining physical context/VRAM;
    # callers may request a smaller budget but there is no service-side
    # arbitrary maximum below the model's own context window.
    max_tokens:int|None=Field(default=None,ge=1)
    temperature:float=Field(default=0.2,ge=0,le=2)
    stream:bool=False
    response_format:dict|None=None
    seed:int|None=Field(default=None,strict=True,ge=0,le=2**63-1)
    chat_template_kwargs:ChatTemplateOptions|None=None
    stop:StopText|list[StopText]|None=None

    @field_validator('stop')
    @classmethod
    def nonempty_stop_sequences(cls,value):
        if isinstance(value,list) and not value:
            raise ValueError('Use at least one nonempty stop sequence')
        return value

class TranscriptionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    audio_base64:str=Field(min_length=1)

ASR_MODEL=ROOT/'optional-models/Whisper-base/base.pt'
ASR_SHA='ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e'

@app.post('/factory/transcribe')
def transcribe(body:TranscriptionRequest):
    model=None
    try:
        raw=base64.b64decode(body.audio_base64,validate=True)
        with wave.open(BytesIO(raw),'rb') as audio:
            if audio.getnchannels()!=1 or audio.getsampwidth()!=2 or audio.getframerate()!=16000 or audio.getnframes() <= 0:
                raise ValueError('Mono 16kHz PCM16 source audio required')
            pcm=audio.readframes(audio.getnframes())
        if not ASR_MODEL.is_file() or digest(ASR_MODEL)!=ASR_SHA:raise ValueError('Verified local Whisper-base is unavailable')
        import numpy as np
        import torch
        import whisper
        if not lock.acquire(blocking=False):raise HTTPException(409,'Local model is busy; transcription not repeated')
        try:
            started=time.monotonic();torch.set_num_threads(4)
            model=whisper.load_model(str(ASR_MODEL),device='cuda')
            result=model.transcribe(np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.0,
                language=None,fp16=True,temperature=0,condition_on_previous_text=False,verbose=None)
            segments=[{'start':float(s['start']),'end':float(s['end']),'text':s['text'],
                'speaker':'unknown','needs_review':True} for s in result.get('segments',[])]
            return {'text':result['text'],'language':result.get('language','zh'),'segments':segments,
                'speaker':'unknown','needs_review':True,
                'receipt':{'model_id':'openai/whisper-base','model_sha256':ASR_SHA,'input_sha256':hashlib.sha256(raw).hexdigest(),
                  'device':torch.cuda.get_device_name(),'elapsed_seconds':round(time.monotonic()-started,3),
                  'script_sha256':SCRIPT_SHA,'offline':True,'speaker_diarization':False}}
        finally:
            model=None;gc.collect();torch.cuda.empty_cache();lock.release()
    except HTTPException:raise
    except (ValueError,wave.Error) as error:raise HTTPException(422,str(error)) from error
    except Exception as error:raise HTTPException(500,{'code':'local_transcription_failed','message':type(error).__name__+': '+str(error)[:1000]}) from error

@app.get('/health')
def health():
    return {'service':SERVICE_ID,'profile':PROFILE,'model_id':MODEL_ID,'workspace':str(ROOT),
            'script_sha256':SCRIPT_SHA,'source_components':SOURCE_COMPONENTS,'pid':os.getpid(),
            'loaded':loaded,'busy':lock.locked() or retained_gpu_lease is not None or loaded,'offline':True,
            'inference_phase':inference_phase,'gpu_release_status':gpu_release_status,
            'gpu_memory':cuda_memory_snapshot(),'last_gpu_cleanup':last_gpu_cleanup,'max_model_len':model_context_limit(),
            'prose_repetition_policy':{'enabled':H3IR_PROSE_REPEAT_GUARD,'policy':'h3ir-output-repetition-v1',
                'scope':'generated_tokens_only','output_repetition_penalty':PROSE_OUTPUT_REPETITION_PENALTY,
                'eos_logits_exempt':True,'min_period_tokens':PROSE_REPEAT_MIN_PERIOD,
                'max_period_tokens':PROSE_REPEAT_MAX_PERIOD,'required_adjacent_repetitions':PROSE_REPEAT_OCCURRENCES,
                'check_every_tokens':PROSE_REPEAT_CHECK_EVERY}}

@app.get('/v1/models')
def models():
    return {'object':'list','data':[{'id':MODEL_ID,'object':'model','owned_by':'local',
       'max_model_len':model_context_limit(),'output_budget':'dynamic_remaining_context_and_vram',
       'input_modalities':['text','image'],'video_adapter':'timestamped extracted frames; no audio transcription'}]}

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):value.update(chunk)
    return value.hexdigest()

def _fixed_model_file(name):
    directory=MODEL.resolve()
    if not isinstance(name,str) or name in {'.','..'} or any(char in name for char in '/\\:'):
        raise ValueError('Model manifest contains a non-local filename')
    path=(directory/name).resolve()
    if path.parent!=directory or not path.is_file():
        raise ValueError('Model file is missing or escapes the fixed model directory: '+name)
    return path


def _model_text_config():
    config=json.loads(_fixed_model_file('config.json').read_text(encoding='utf-8-sig')).get('text_config')
    if not isinstance(config,dict):
        raise ValueError('Fixed local model config has no text_config')
    return config


def model_context_limit(config=None):
    """Use the checkpoint context unless a wrapper declares a physical limit."""
    config=_model_text_config() if config is None else config
    advertised=config.get('max_position_embeddings')
    if type(advertised) is not int or advertised <= 1:
        raise ValueError('Fixed local model has no usable max_position_embeddings')
    configured=MAX_CONTEXT_TOKENS
    if configured is None:
        return advertised
    if type(configured) is not int or configured <= 1:
        raise ValueError('Configured local context limit is invalid')
    return min(configured,advertised)

def verify_model_weights():
    """Verify fixed on-disk weights; manifest.directory is informational only."""
    manifest_path=_fixed_model_file('factory-model-manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('repository')!=MODEL_ID:
        raise ValueError('Local model manifest repository does not match the fixed profile')
    if 'weights_sha256' in manifest:
        expected={'model.safetensors':{'sha256':manifest['weights_sha256'],'bytes':manifest.get('weights_bytes')}}
    elif isinstance(manifest.get('weight_files'),dict) and manifest['weight_files']:
        metadata=json.loads(_fixed_model_file('official-repository-metadata.json').read_text(encoding='utf-8-sig'))
        records={item.get('Path'):item for item in metadata.get('Data',{}).get('Files',[]) if isinstance(item,dict)}
        expected={}
        for name,sha in manifest['weight_files'].items():
            _fixed_model_file(name)
            record=records.get(name,{})
            if record.get('Sha256')!=sha or record.get('Revision')!=manifest.get('revision'):
                raise ValueError('Shard manifest and saved official metadata disagree: '+name)
            if type(record.get('Size')) is not int or record['Size']<=0:
                raise ValueError('Shard lacks its expected byte length: '+name)
            expected[name]={'sha256':sha,'bytes':record['Size']}
        index_path=_fixed_model_file('model.safetensors.index.json')
        index_record=records.get(index_path.name,{})
        if index_record.get('Size')!=index_path.stat().st_size or index_record.get('Sha256')!=digest(index_path):
            raise ValueError('Saved model shard index bytes or SHA256 changed')
        index=json.loads(index_path.read_text(encoding='utf-8-sig'))
        if not isinstance(index.get('weight_map'),dict) or set(index['weight_map'].values())!=set(expected):
            raise ValueError('Model shard index references unverified or missing shard files')
    else:
        raise ValueError('Local model manifest has no supported weight hashes')
    verified=[]
    for name,record in expected.items():
        if not name.endswith('.safetensors') or not isinstance(record['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',record['sha256']):
            raise ValueError('Invalid model weight hash or filename')
        path=_fixed_model_file(name)
        before=path.stat()
        if record['bytes'] is not None and (type(record['bytes']) is not int or before.st_size!=record['bytes']):
            raise ValueError('Model weight byte length changed: '+name)
        actual=digest(path)
        after=path.stat()
        if actual!=record['sha256'] or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Model weight SHA256 changed: '+name)
        verified.append({'name':name,'bytes':after.st_size,'expected_bytes':record['bytes'],'sha256':actual})
    return {'repository':MODEL_ID,'revision':manifest.get('revision'),
            'manifest_sha256':digest(manifest_path),'files':verified,
            'total_weight_file_bytes':sum(item['bytes'] for item in verified),'source':'fixed_local_directory'}

def acquire_gpu_lease():
    """Share one GPU inference slot across the 2B and 4B local processes."""
    GPU_LEASE_PATH.parent.mkdir(parents=True,exist_ok=True)
    lease=GPU_LEASE_PATH.open('a+b')
    try:
        lease.seek(0,2)
        if lease.tell()==0:lease.write(b'\0');lease.flush()
        lease.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lease.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return lease
    except OSError as error:
        lease.close()
        raise HTTPException(409,'Another local Qwen inference holds the GPU slot; request was not repeated') from error

def release_gpu_lease(lease):
    if lease is not None:
        try:
            lease.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(lease.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN)
        finally:lease.close()

def output_token_budget(torch,prompt_tokens,requested_output_tokens):
    """Select the remaining context and VRAM capacity when no output limit is requested."""
    automatic=requested_output_tokens is None
    check_vram=automatic or VRAM_PREFLIGHT
    config=_model_text_config()
    context_limit=model_context_limit(config)
    available_context=context_limit-prompt_tokens
    output_tokens=available_context if automatic else requested_output_tokens
    if not 0<output_tokens<=available_context:
        raise ValueError(f'Input/output exceeds the local {context_limit}-token context budget; reduce or split the request')
    if not check_vram:return output_tokens,None
    head_dim=config.get('head_dim') or config['hidden_size']//config['num_attention_heads']
    kv_bytes_per_token=2*2*config['num_hidden_layers']*config['num_key_value_heads']*head_dim
    weight_bytes=weights_receipt['total_weight_file_bytes']
    headroom=1024**3
    free,total=torch.cuda.mem_get_info()
    if automatic:
        available_vram=(free-weight_bytes-headroom)//kv_bytes_per_token-prompt_tokens
        output_tokens=min(output_tokens,available_vram)
    kv_bytes=kv_bytes_per_token*(prompt_tokens+max(output_tokens,1))
    estimate=weight_bytes+kv_bytes+headroom
    if free<estimate:
        raise ValueError(f'Local VRAM preflight requires an estimated {estimate} free bytes but only {free} are available; no model was loaded')
    return output_tokens,{'free_bytes':free,'total_bytes':total,'estimated_weights_bytes':weight_bytes,
            'estimated_kv_cache_bytes':kv_bytes,'headroom_bytes':headroom,
            'result':'estimated_budget_available','guarantees_inference_success':False}

def efficient_text_sdpa(module,query,key,value,attention_mask,dropout=0.0,scaling=None,is_causal=None,**kwargs):
    """Windows-safe text attention: explicit GQA expansion and no math fallback."""
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    if kwargs.get('output_attentions',False) or kwargs.get('head_mask') is not None:
        raise ValueError('Safe text SDPA does not support attention-weight output or head masks')
    if query.shape[1]!=key.shape[1]:
        if key.shape[1]!=value.shape[1] or query.shape[1]%key.shape[1]:
            raise ValueError('Unsupported GQA head dimensions')
        groups=query.shape[1]//key.shape[1]
        def expand_kv(tensor):
            batch,heads,length,dimension=tensor.shape
            return tensor[:,:,None,:,:].expand(batch,heads,groups,length,dimension).reshape(batch,heads*groups,length,dimension)
        key,value=expand_kv(key),expand_kv(value)
    if attention_mask is not None and attention_mask.ndim==4:
        attention_mask=attention_mask[:,:,:,:key.shape[-2]]
    if is_causal is None:
        is_causal=query.shape[2]>1 and attention_mask is None and getattr(module,'is_causal',True)
    # Native GQA makes HF 4.57 request enable_gqa=True when no mask is present.
    # Windows efficient SDPA rejects that path; default dispatch can then choose
    # an N*N fp32 math matrix. Repeated KV keeps the efficient backend eligible.
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        output=torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=attention_mask,
            dropout_p=dropout,scale=scaling,is_causal=is_causal,enable_gqa=False)
    return output.transpose(1,2).contiguous(),None

def configure_model_attention(model):
    if not SAFE_TEXT_SDPA:
        return {'safe_text_sdpa':False,'requested':'sdpa'}
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    name='video_factory_efficient_text_sdpa'
    ALL_ATTENTION_FUNCTIONS.register(name,efficient_text_sdpa)
    model.set_attn_implementation({'text_config':name,'vision_config':'sdpa'})
    text=[item for _,item in model.named_modules() if type(item).__name__=='Qwen3VLTextAttention']
    if not text or any(item.config._attn_implementation!=name for item in text):
        raise ValueError('Safe text attention interface was not installed on every Qwen3-VL text layer')
    return {'safe_text_sdpa':True,'text_implementation':name,'text_layers':len(text),
            'gqa':'explicit_repeat_kv','allowed_text_backend':'EFFICIENT_ATTENTION',
            'math_fallback':False,'vision_implementation':model.config.vision_config._attn_implementation}

def cuda_memory_snapshot():
    torch=sys.modules.get('torch')
    cuda=getattr(torch,'cuda',None)
    if cuda is None or not getattr(cuda,'is_initialized',lambda:False)():
        return {'state':'not_initialized','allocated_bytes':0,'reserved_bytes':0}
    try:
        free,total=cuda.mem_get_info()
        return {'state':'ok','allocated_bytes':cuda.memory_allocated(),'reserved_bytes':cuda.memory_reserved(),
                'global_free_bytes':free,'global_total_bytes':total}
    except Exception as error:
        return {'state':'unknown','error':type(error).__name__+': '+str(error)[:300]}

def discard_exception_frames(error):
    """Drop model/activation references before releasing the cross-process slot."""
    pending=[error];seen=set()
    while pending:
        item=pending.pop()
        if id(item) in seen:continue
        seen.add(id(item))
        pending.extend(child for child in (item.__cause__,item.__context__) if isinstance(child,BaseException))
        if item.__traceback__ is not None:traceback.clear_frames(item.__traceback__)
        item.__traceback__=None;item.__cause__=None;item.__context__=None
        item.args=()
        if isinstance(getattr(item,'__dict__',None),dict):item.__dict__.clear()

def exception_http_details(error):
    if isinstance(error,HTTPException):
        # Only JSON-safe details cross this boundary, never the exception object.
        return error.status_code,json.loads(json.dumps(error.detail,default=str)),dict(error.headers or {})
    if isinstance(error,(ValueError,UnidentifiedImageError)):return 422,str(error)[:1200],{}
    return 500,{'code':'local_inference_failed','message':type(error).__name__+': '+str(error)[:1200]},{}

def cleanup_gpu(torch,baseline):
    gc.collect()
    cleanup_error=None
    try:torch.cuda.empty_cache()
    except Exception as error:
        cleanup_error=type(error).__name__+': '+str(error)[:300]
        discard_exception_frames(error)
    current=cuda_memory_snapshot()
    released=current['state']=='not_initialized'
    if current['state']=='ok':
        released=current['allocated_bytes']<=baseline.get('allocated_bytes',0)+16*1024**2
        if baseline.get('state')=='ok':
            released=released and current['global_free_bytes']>=baseline['global_free_bytes']-256*1024**2
    if cleanup_error:released=False
    return {'released':bool(released),'baseline':baseline,'after':current,'error':cleanup_error,
            'scope':'own live Torch allocation plus conservative global-free comparison; not peak memory'}

def _read_memory_value(path):
    try:
        raw=Path(path).read_text(encoding='ascii').strip()
    except OSError:
        return None
    if raw.casefold()=='max':return None
    try:
        value=int(raw)
    except ValueError:
        return None
    return value if value>=0 else None


def cgroup_available_memory():
    """Return actual cgroup headroom when the current host exposes it."""
    candidates=(
        ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    )
    values=[]
    for limit_path,current_path in candidates:
        limit,current=_read_memory_value(limit_path),_read_memory_value(current_path)
        if limit is not None and current is not None:
            # An over-limit cgroup has zero usable headroom, not an unknown
            # limit.  Reporting zero lets the dynamic preflight produce the
            # real physical-memory diagnostic before another allocation.
            values.append(max(0,limit-current))
    return min(values) if values else None


def available_host_memory():
    """Measure available RAM, respecting both host and container accounting."""
    import psutil
    process=psutil.Process()
    host_available=int(psutil.virtual_memory().available)
    cgroup_available=cgroup_available_memory()
    usable=min(value for value in (host_available,cgroup_available) if value is not None)
    return {'host_available_bytes':host_available,'cgroup_available_bytes':cgroup_available,
            'available_bytes':usable,'process_rss_bytes':int(process.memory_info().rss)}


def checkpoint_max_pixels(image_processor):
    """Use only a value saved by this checkpoint's native image processor."""
    try:
        saved=image_processor.to_dict()
    except Exception:
        return None
    value=saved.get('max_pixels') if isinstance(saved,dict) else None
    return value if type(value) is int and value>0 and getattr(image_processor,'max_pixels',value)==value else None


def header_decode_bytes(image):
    """Estimate the source frame allocation from its real PIL header mode."""
    mode=str(image.mode or '')
    bits_per_channel=32 if mode in {'I','F'} else 16 if mode.startswith('I;16') else 8
    channels=max(1,len(image.getbands() or ()))
    return int(image.width)*int(image.height)*channels*((bits_per_channel+7)//8)


def preflight_images_for_processor(images,image_records,image_processor):
    """Apply checkpoint-native JPEG drafting and dynamic RAM admission only.

    There is deliberately no image count, byte, edge, or service pixel ceiling:
    the estimate is derived from each real header and the memory available at
    this exact moment.  Non-JPEG sources remain untouched because Pillow has no
    equivalent safe native draft path for them.
    """
    native_max_pixels=checkpoint_max_pixels(image_processor)
    for image,record in zip(images,image_records):
        original_dimensions=[int(image.width),int(image.height)]
        drafted=False
        if (native_max_pixels is not None and (image.format or '').upper()=='JPEG'
                and image.width*image.height>native_max_pixels):
            scale=(native_max_pixels/(image.width*image.height))**0.5
            target=(max(1,int(image.width*scale)),max(1,int(image.height*scale)))
            try:
                image.draft('RGB',target)
                drafted=True
            except (OSError,ValueError):
                # The original remains valid and is protected by the same
                # dynamic preflight below.  Do not silently discard it.
                drafted=False
        record['header_dimensions']=original_dimensions
        record['effective_header_dimensions']=[int(image.width),int(image.height)]
        record['native_jpeg_draft_applied']=drafted

    # Each image can coexist as a decoded RGB frame, an EXIF-orientation copy,
    # a processor float working array and an overlapping conversion buffer.
    # This is an allocation estimate, not a static image limit; it scales with
    # the exact inputs.
    estimated_bytes=0
    for image in images:
        pixels=int(image.width)*int(image.height)
        decoded_rgb=pixels*3
        orientation_rgb=pixels*3
        processor_float_rgb=pixels*3*4
        estimated_bytes+=(header_decode_bytes(image)+decoded_rgb+orientation_rgb
                          +processor_float_rgb+max(decoded_rgb,orientation_rgb,processor_float_rgb))
    availability=available_host_memory()
    receipt={'estimated_additional_decoder_processor_bytes':estimated_bytes,
             'checkpoint_native_max_pixels':native_max_pixels,
             **availability}
    if estimated_bytes>availability['available_bytes']:
        raise ValueError('Physical host memory preflight has insufficient memory for native image decoding: '
                         f'estimated_additional_bytes={estimated_bytes}, actual_available_bytes={availability["available_bytes"]}')
    return receipt


def apply_exif_orientation(images,messages,image_records):
    """Orient frames after RAM admission and keep message/image identities aligned."""
    replacements={}
    for index,image in enumerate(images):
        try:orientation=image.getexif().get(0x0112)
        except Exception:orientation=None
        try:
            # Modern Pillow can retain the same Image object, avoiding a stale
            # reference in the chat message parts.
            oriented=ImageOps.exif_transpose(image,in_place=True)
            oriented=image if oriented is None else oriented
        except TypeError:
            # Older supported Pillow releases return a new object instead.
            oriented=ImageOps.exif_transpose(image)
        if oriented is not image:
            images[index]=oriented
            replacements[id(image)]=oriented
            try:image.close()
            except Exception:pass
        image_records[index]['exif_orientation']=orientation
        image_records[index]['exif_orientation_applied']=orientation in {2,3,4,5,6,7,8}
    if replacements:
        for message in messages:
            content=message.get('content')
            if not isinstance(content,list):continue
            for part in content:
                original=part.get('image') if isinstance(part,dict) and part.get('type')=='image' else None
                if id(original) in replacements:part['image']=replacements[id(original)]


def preflight_encoded_image_decode(encoded_characters):
    """Reserve only the RAM implied by this actual inline Base64 payload.

    The request body already owns the encoded string.  Decoding it can briefly
    require a byte buffer plus the BytesIO backing store, so measure that
    dynamic allocation against current host/cgroup headroom rather than using a
    fixed file-size policy.  Strict payload validation remains with
    ``b64decode(..., validate=True)`` below.
    """
    # A valid Base64 quartet yields at most three bytes.  Rounding up is a
    # conservative allocation estimate for malformed input before the strict
    # decoder supplies its normal format error.
    decoded_upper_bound=((encoded_characters+3)//4)*3
    # Slicing the request's URL creates one encoded-string copy and the strict
    # decoder may make an ASCII byte copy before allocating the decoded bytes
    # and BytesIO backing store.  Count those real transient allocations while
    # keeping the admission decision entirely dynamic.
    estimated_additional_bytes=encoded_characters*2+decoded_upper_bound*2
    availability=available_host_memory()
    receipt={'encoded_characters':encoded_characters,
             'estimated_additional_decode_bytes':estimated_additional_bytes,
             **availability}
    if estimated_additional_bytes>availability['available_bytes']:
        raise ValueError('Physical host memory preflight has insufficient memory for uploaded image decoding: '
                         f'estimated_additional_bytes={estimated_additional_bytes}, actual_available_bytes={availability["available_bytes"]}')
    return receipt


def close_prepared_images(images=(),image_sources=()):
    """Release lazy Pillow images and their backing buffers on every exit path."""
    for image in images or ():
        try:image.close()
        except Exception:pass
    for source in image_sources or ():
        try:source.close()
        except Exception:pass


def prepare(messages):
    converted=[];images=[];image_sources=[];image_records=[];chars=0
    try:
        # Do not decode two potentially large Base64 payload sets in parallel.
        # This is admission serialization only; it introduces no size/count or
        # wall-clock policy limit.
        with input_decode_lock:
            for message in messages:
                if set(message)-{'role','content'} or message.get('role') not in {'system','user','assistant'}:raise ValueError('Unsupported chat message fields')
                content=message.get('content')
                if isinstance(content,str):chars+=len(content);converted.append({'role':message['role'],'content':content});continue
                if not isinstance(content,list):raise ValueError('Content must be text or typed text/image parts')
                parts=[]
                for part in content:
                    if part.get('type')=='text' and isinstance(part.get('text'),str):
                        chars+=len(part['text']);parts.append({'type':'text','text':part['text']})
                    elif part.get('type')=='image_url':
                        url=part.get('image_url',{}).get('url','')
                        if not isinstance(url,str) or not url.startswith(('data:image/png;base64,','data:image/jpeg;base64,','data:image/webp;base64,')):raise ValueError('Only uploaded inline image bytes are accepted; URLs and filesystem paths are forbidden')
                        delimiter=url.find(',')
                        decode_preflight=preflight_encoded_image_decode(len(url)-delimiter-1)
                        payload=url[delimiter+1:]
                        raw=base64.b64decode(payload,validate=True)
                        raw_sha256=hashlib.sha256(raw).hexdigest()
                        raw_bytes=len(raw)
                        # Header parsing establishes that this is an image without
                        # eagerly decompressing or converting a potentially huge
                        # compressed source.  Keep the source buffer alive until the
                        # Qwen processor performs its native, model-aligned resize and
                        # token accounting below.
                        source=BytesIO(raw)
                        del raw
                        try:image=Image.open(source)
                        except BaseException:
                            source.close()
                            raise
                        # Register each resource before any later metadata work can
                        # fail, so the outer cleanup covers every partial image.
                        image_sources.append(source);images.append(image)
                        image_records.append({'image_ordinal':len(image_records)+1,
                            'sha256':raw_sha256,'bytes':raw_bytes,
                            'encoded_decode_memory_preflight':decode_preflight,
                            'format':image.format,'mode':image.mode,
                            'header_dimensions':[int(image.width),int(image.height)]})
                        parts.append({'type':'image','image':image})
                    else:raise ValueError('Unsupported multimodal content part')
                converted.append({'role':message['role'],'content':parts})
        return converted,images,image_sources,image_records
    except BaseException:
        close_prepared_images(images,image_sources)
        raise

def request_controls(body):
    # This fixed checkpoint is Instruct, not its Thinking sibling. A template
    # flag cannot turn it into a different model, so do not silently accept True.
    if body.chat_template_kwargs is not None and body.chat_template_kwargs.enable_thinking:
        raise ValueError('Qwen3-VL-2B-Instruct does not support enable_thinking=true; use false or omit it')
    return body.seed if body.seed is not None else 1247

def template_controls(body,processor):
    template=getattr(processor,'chat_template',None)
    if template is None:template=getattr(getattr(processor,'tokenizer',None),'chat_template',None)
    if isinstance(template,dict):template=template.get('default')
    if not isinstance(template,str):raise ValueError('The actual local chat template cannot be verified')
    requested=body.chat_template_kwargs.enable_thinking if body.chat_template_kwargs is not None else None
    if 'enable_thinking' in template:
        # Pass a supported control to the real template, rather than merely
        # accepting it in the OpenAI-compatible request schema.
        kwargs={'enable_thinking':False}
        mode='template_disabled'
    else:
        if '<think>' in template or 'reasoning_content' in template:
            raise ValueError('Actual chat template has unsupported thinking behavior; request not ignored')
        kwargs={}
        mode='fixed_instruct_non_thinking'
    return kwargs,{'requested_enable_thinking':requested,'effective_enable_thinking':False,
                   'mode':mode,'template_sha256':hashlib.sha256(template.encode()).hexdigest()}

def generation_stop_controls(body,processor):
    if body.stop is None:return {},[]
    stops=[body.stop] if isinstance(body.stop,str) else body.stop
    tokenizer=getattr(processor,'tokenizer',None)
    if tokenizer is None:raise ValueError('Stop strings require the actual model tokenizer')
    # Installed Transformers implements this as StopStringCriteria during
    # decoding, including stops that span token boundaries.
    return {'stop_strings':stops,'tokenizer':tokenizer},stops

def generation_sampling_controls(body,model):
    if body.temperature==0:
        # Clear inherited sampling-only defaults as well, so Transformers does
        # not warn that the checkpoint's temperature/top-p/top-k were ignored.
        return {'do_sample':False,'temperature':None,'top_p':None,'top_k':None}
    return {'do_sample':True,'temperature':body.temperature,
            'top_p':getattr(model.generation_config,'top_p',1.0),
            'top_k':getattr(model.generation_config,'top_k',50)}

class OutputOnlyRepetitionPenalty:
    """Use the official HF penalty only on emitted tokens; leave EOS untouched."""
    def __init__(self,prompt_tokens,penalty=1.1,eos_ids=(),processor=None):
        if type(prompt_tokens) is not int or prompt_tokens<0 or not 1.0<float(penalty)<=1.2:
            raise ValueError('Invalid output repetition penalty policy')
        self.prompt_tokens=prompt_tokens;self.penalty=float(penalty)
        self.eos_ids=tuple(sorted(set(int(token) for token in eos_ids if token is not None)))
        self.uses_prompt_ignore_length=False
        if processor is None:
            import inspect
            from transformers import RepetitionPenaltyLogitsProcessor
            self.uses_prompt_ignore_length='prompt_ignore_length' in inspect.signature(RepetitionPenaltyLogitsProcessor).parameters
            options={'prompt_ignore_length':prompt_tokens} if self.uses_prompt_ignore_length else {}
            processor=RepetitionPenaltyLogitsProcessor(penalty=self.penalty,**options)
        self.processor=processor

    def __call__(self,input_ids,scores):
        if input_ids.shape[-1]<=self.prompt_tokens:return scores
        eos=[token for token in self.eos_ids if 0<=token<scores.shape[-1]]
        original_eos=scores[:,eos].clone() if eos else None
        emitted=input_ids if self.uses_prompt_ignore_length else input_ids[:,self.prompt_tokens:]
        result=self.processor(emitted,scores)
        if eos:result[:,eos]=original_eos
        return result


class RepeatedOutputGuard:
    """Stop an adjacent long token-block loop, without counting the prompt.

    This does not finish, repair or trim a brief. A trigger remains an explicit
    incomplete-generation failure. Short chants and non-adjacent repetitions
    (for example a name in several required sections) do not qualify.
    """
    def __init__(self,prompt_tokens,eos_ids=(),*,min_period=32,max_period=256,repetitions=3,check_every=8):
        if (type(prompt_tokens) is not int or prompt_tokens<0 or not 16<=min_period<=max_period<=512
                or not 3<=repetitions<=4 or not 1<=check_every<=32):
            raise ValueError('Invalid bounded prose repetition guard')
        self.prompt_tokens=prompt_tokens;self.eos_ids=set(eos_ids)
        self.min_period=min_period;self.max_period=max_period
        self.repetitions=repetitions;self.check_every=check_every
        self.triggered=False;self.checked_tokens=0;self.period_tokens=None

    def __call__(self,input_ids,scores=None,**kwargs):
        shape=getattr(input_ids,'shape',None)
        length=(int(shape[-1]) if shape is not None else len(input_ids[0]))-self.prompt_tokens
        if self.triggered:return True
        if length<self.min_period*self.repetitions or length%self.check_every or length==self.checked_tokens:return False
        if (int(shape[0]) if shape is not None else len(input_ids))!=1:
            raise ValueError('The fixed prose guard requires a single generated sequence')
        row=input_ids[0];tail=row[-min(length,self.max_period*self.repetitions):]
        tail=tail.tolist() if hasattr(tail,'tolist') else list(tail)
        self.checked_tokens=length
        if tail and tail[-1] in self.eos_ids:return False
        for period in range(self.min_period,min(self.max_period,length//self.repetitions)+1):
            block=tail[-period:]
            if block*self.repetitions!=tail[-period*self.repetitions:]:continue
            # A long composite of a short chant is still a short chant.
            if any(period%unit==0 and block==block[:unit]*(period//unit) for unit in range(1,self.min_period)):continue
            self.triggered=True;self.period_tokens=period
            return True
        return False

    def receipt(self):
        return {'scope':'generated_tokens_only','triggered':self.triggered,
                'min_period_tokens':self.min_period,'max_period_tokens':self.max_period,
                'required_adjacent_repetitions':self.repetitions,'check_every_tokens':self.check_every,
                'checked_generated_tokens':self.checked_tokens,'matched_period_tokens':self.period_tokens,
                'completion_rewritten':False,'incomplete_finish_reason':'repetition_limit'}


def generation_prose_controls(body,model,prompt_tokens,*,existing_stopping=None):
    enabled=H3IR_PROSE_REPEAT_GUARD and body.response_format is None
    receipt={'enabled':bool(enabled),'policy':'h3ir-output-repetition-v1','scope':'generated_tokens_only'}
    if not enabled:return {},receipt,None
    from transformers import LogitsProcessorList,StoppingCriteriaList
    eos=model.generation_config.eos_token_id
    eos_ids=eos if isinstance(eos,(list,tuple)) else [eos]
    penalty=OutputOnlyRepetitionPenalty(prompt_tokens,PROSE_OUTPUT_REPETITION_PENALTY,eos_ids)
    guard=RepeatedOutputGuard(prompt_tokens,eos_ids,min_period=PROSE_REPEAT_MIN_PERIOD,
        max_period=PROSE_REPEAT_MAX_PERIOD,repetitions=PROSE_REPEAT_OCCURRENCES,check_every=PROSE_REPEAT_CHECK_EVERY)
    receipt.update(output_repetition_penalty=penalty.penalty,eos_logits_exempt=True,
                   hf_prompt_ignore_length=penalty.uses_prompt_ignore_length,
                   inherited_prompt_penalty_disabled=True)
    return {'repetition_penalty':1.0,'logits_processor':LogitsProcessorList([penalty]),
            'stopping_criteria':StoppingCriteriaList([*(existing_stopping or []),guard])},receipt,guard


def generation_finish_reason(ended,matched_stop,generated_tokens,max_tokens,guard=None):
    if guard is not None and guard.triggered:return 'repetition_limit'
    return 'stop' if ended or matched_stop is not None else 'length' if generated_tokens>=max_tokens else 'time_limit'

def trim_stop_output(result,raw_result,stops):
    matches=[(raw_result.find(stop),index,stop) for index,stop in enumerate(stops) if stop in raw_result]
    if matches:
        _,_,matched=min(matches)
        # Special stop tokens may already have been removed by skip_special_tokens.
        visible=[result.find(stop) for stop in stops if stop in result]
        return result[:min(visible)] if visible else result,matched
    # Transformers can correctly stop a literal OCR response at a
    # prompt/generation boundary (the prompt supplies the preceding newline),
    # while the decoded generated suffix contains only ``[unclear]``. That is
    # a complete, truthful OCR result, not a timeout. Preserve the literal
    # value and record it as a semantic stop; no output-size fallback is used.
    if (raw_result.strip()=='[unclear]'
            and any(stop.strip()=='[unclear]' for stop in stops)):
        return result,'[unclear]'
    return result,None

def _strict_json_loads(text):
    def unique_object(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON object key: '+key)
            result[key]=value
        return result
    def invalid_constant(value):raise ValueError('Non-JSON numeric constant: '+value)
    return json.loads(text,object_pairs_hook=unique_object,parse_constant=invalid_constant)

def _check_guided_schema(schema,depth=0):
    # Do not let a permissive parser silently ignore unsupported JSON Schema
    # structures or let the post-validator resolve anything over the network.
    if not isinstance(schema,dict) or depth>32:raise ValueError('Unsupported or deeply nested JSON schema')
    allowed={'type','properties','required','additionalProperties','items','enum','const','anyOf',
             '$ref','$defs','definitions','$schema','$id','title','description','default','examples',
             'minLength','maxLength','minItems','maxItems'}
    unknown=set(schema)-allowed
    if unknown:raise ValueError('Unsupported guided JSON schema keyword(s): '+', '.join(sorted(unknown)))
    if '$ref' in schema and not (isinstance(schema['$ref'],str) and re.fullmatch(r'#/(?:\$defs|definitions)/[^/~]+',schema['$ref'])):
        raise ValueError('Only local $defs/definitions references are supported; remote references are forbidden')
    if isinstance(schema.get('type'),list):raise ValueError('Use explicit anyOf schemas for nullable values')
    if 'const' in schema and not isinstance(schema['const'],str):
        raise ValueError('This guided backend supports string const; use typed structures for other constants')
    if 'enum' in schema:
        values=schema['enum']
        if not isinstance(values,list) or not values or not (all(isinstance(v,str) for v in values) or all(type(v) in (int,float) for v in values)):
            raise ValueError('Guided enum values must be uniformly strings or numbers')
    for key in ('properties','$defs','definitions'):
        if key in schema:
            if not isinstance(schema[key],dict):raise ValueError('Invalid guided schema '+key)
            for child in schema[key].values():_check_guided_schema(child,depth+1)
    if 'items' in schema:_check_guided_schema(schema['items'],depth+1)
    if isinstance(schema.get('additionalProperties'),dict):_check_guided_schema(schema['additionalProperties'],depth+1)
    if 'anyOf' in schema:
        if not isinstance(schema['anyOf'],list) or not schema['anyOf']:raise ValueError('Invalid guided anyOf')
        for child in schema['anyOf']:_check_guided_schema(child,depth+1)

def prepare_response_format(body):
    requested=body.response_format
    if requested is None:return None
    if body.stop is not None:raise ValueError('Explicit stop strings are unsupported with guided JSON; completion must emit real EOS')
    kind=requested.get('type')
    if kind=='json_object':
        if set(requested)!={'type'}:raise ValueError('json_object accepts only the type field')
        schema={'type':'object'}
    elif kind=='json_schema':
        if set(requested)!={'type','json_schema'}:raise ValueError('Invalid json_schema response_format fields')
        details=requested.get('json_schema')
        if not isinstance(details,dict) or not {'name','schema','strict'}<=set(details) or set(details)-{'name','schema','strict','description'}:
            raise ValueError('json_schema requires name, schema and strict=true')
        if details['strict'] is not True:raise ValueError('Only strict=true guided JSON schema requests are supported')
        if not isinstance(details['name'],str) or not details['name'].strip():raise ValueError('json_schema name must be a nonempty string')
        schema=details['schema']
        if not isinstance(schema,dict) or schema.get('type')!='object':raise ValueError('Guided response schema must have a top-level object type')
    else:raise ValueError('Supported response_format types are json_object and json_schema')
    _check_guided_schema(schema)
    try:
        from jsonschema.validators import validator_for
    except ImportError as error:
        raise HTTPException(503,'Guided JSON requires the locally installed jsonschema validator; no unguarded fallback') from error
    try:
        validator_class=validator_for(schema)
        validator_class.check_schema(schema)
        validator=validator_class(schema)
    except Exception as error:raise ValueError('Invalid JSON schema: '+str(error)[:500]) from error
    encoded=json.dumps(schema,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return {'schema':schema,'validator':validator,'receipt':{'requested':kind,'guided_decoding':True,
            'schema_sha256':hashlib.sha256(encoded).hexdigest(),'backend':'lm-format-enforcer',
            'post_validation':False,'semantic_correctness_guaranteed':False}}

def _install_json_literal_grammar():
    """Encode fixed JSON strings as literals, including their escape sequences.

    lm-format-enforcer 0.11.3's StringParsingState feeds raw const/enum values
    to its character parser. A quote within a fixed dialogue then demands an
    invalid unescaped quote, and rejects the valid JSON backslash. Use the
    library's literal parser for only those fixed values. The schema and
    decoded values stay unchanged; original strict post-validation still runs.
    No vendor files, ordinary free strings or object keys are modified.
    """
    import lmformatenforcer.jsonschemaparser as module
    from lmformatenforcer import StringParser,UnionParser,SequenceParser,RegexParser
    original=module.get_parser
    if getattr(original,'_factory_escaped_json_literals',False):return
    @functools.wraps(original)
    def fixed_literals(root,value_schema):
        extras=value_schema.extras or {}
        values=([extras['const']] if isinstance(extras.get('const'),str)
                else value_schema.enum if value_schema.type=='string' else None)
        if values and all(isinstance(value,str) for value in values):
            # JSON permits whitespace before a value, but not within a fixed
            # literal. Keep the two states separate, including leading spaces
            # that are part of the caller's actual string value.
            parsers=[SequenceParser([RegexParser('[ \\t\\r\\n]*',root.config),
                        StringParser(json.dumps(value,ensure_ascii=False))]) for value in values]
            return parsers[0] if len(parsers)==1 else UnionParser(parsers)
        return original(root,value_schema)
    fixed_literals._factory_escaped_json_literals=True
    module.get_parser=fixed_literals


def _build_grammar_prefix(tokenizer,schema):
    global grammar_tokenizer_data,grammar_control_fragments
    try:
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (build_token_enforcer_tokenizer_data,
                                                               build_transformers_prefix_allowed_tokens_fn)
    except ImportError as error:
        raise HTTPException(503,'Guided JSON backend is unavailable; no unguarded fallback') from error
    _install_json_literal_grammar()
    if grammar_tokenizer_data is None:
        grammar_tokenizer_data=build_token_enforcer_tokenizer_data(tokenizer)
        grammar_control_fragments={token:fragment for token,fragment,_ in grammar_tokenizer_data.regular_tokens
                                  if re.search(r'[\x00-\x1f]',fragment)}
    # JsonSchemaParser captures the alphabet for required nonempty strings at
    # construction. Setting it later leaves that branch restricted to ASCII.
    parser_config=CharacterLevelParserConfig(alphabet=grammar_tokenizer_data.tokenizer_alphabet)
    prefix=build_transformers_prefix_allowed_tokens_fn(grammar_tokenizer_data,JsonSchemaParser(schema,parser_config))
    return prefix,grammar_control_fragments


def _json_lexical_state(text,in_string=False,escaped=False):
    for character in text:
        if ord(character)<32 and (in_string or character not in '\t\n\r'):
            return None
        if escaped:
            escaped=False
        elif in_string and character=='\\':
            escaped=True
        elif character=='"':
            in_string=not in_string
    return in_string,escaped

def guided_generation_controls(prepared,processor,model,prompt_tokens):
    if prepared is None:return {},None
    tokenizer=getattr(processor,'tokenizer',None)
    if tokenizer is None:raise ValueError('Guided JSON requires the actual model tokenizer')
    grammar_prefix,control_fragments=_build_grammar_prefix(tokenizer,prepared['schema'])
    eos=model.generation_config.eos_token_id
    eos_ids=set(eos if isinstance(eos,list) else [eos])
    eos_ids={value for value in eos_ids if type(value) is int and value>=0}
    if not eos_ids:raise ValueError('Cannot verify real model EOS for guided JSON')
    stats={'prefix_calls':0,'prompt_tokens_excluded':prompt_tokens,'strict_json_control_character_filter':True,
           'escaped_fixed_string_grammar':'json-literal-v1','original_schema_post_validation':True}
    def prefix(batch_id,sent):
        stats['prefix_calls']+=1
        generated=sent[prompt_tokens:]
        ids=generated.tolist()
        text=tokenizer.decode(ids,skip_special_tokens=False) if ids else ''
        if text.rstrip().endswith('}'):
            try:
                instance=_strict_json_loads(text)
                complete=isinstance(instance,dict) and prepared['validator'].is_valid(instance)
            except (ValueError,TypeError):complete=False
            if complete:return sorted(eos_ids)
        state=_json_lexical_state(text)
        if state is None:
            raise ValueError('Guided JSON prefix contains an invalid unescaped control character')
        # The free-text fast path of lm-format-enforcer can admit a token
        # containing a literal newline inside a string. Mask those tokens
        # before sampling while retaining JSON whitespace outside strings.
        allowed=[token for token in grammar_prefix(batch_id,generated) if token not in eos_ids
                 and (token not in control_fragments or _json_lexical_state(control_fragments[token],*state) is not None)]
        if not allowed:raise ValueError('Guided JSON parser has no valid continuation before a complete instance; no EOS fallback')
        return allowed
    return {'prefix_allowed_tokens_fn':prefix},stats

def validate_guided_output(result,prepared,stats,ended,finish,completion_tokens):
    if prepared is None:return {'requested':None,'guided_decoding':False,'post_validation':False}
    if stats is None or stats['prefix_calls']==0:raise ValueError('Model did not execute the guided token filter')
    if not ended:raise ValueError(f'Guided JSON did not emit real EOS (finish_reason={finish}, completion_tokens={completion_tokens})')
    try:
        instance=_strict_json_loads(result)
        if not isinstance(instance,dict):raise ValueError('Expected a JSON object instance')
        prepared['validator'].validate(instance)
    except Exception as error:raise ValueError('Guided JSON post-validation failed: '+str(error)[:500]) from error
    return {**prepared['receipt'],**stats,'post_validation':True,'completion_tokens':completion_tokens}

def schema_conditioned_messages(messages, prepared):
    """Make the API output contract visible to a plain Transformers model.

    A token mask enforces JSON syntax, but does not teach the model what unseen
    field names, enums or array lengths mean. Keep the schema in the prompt as
    well as in the decoder. Preserve all caller text and actual image parts.
    """
    if prepared is None:
        return messages
    result = [dict(message) for message in messages]
    instruction = ('\n\nReturn ONE JSON data object matching the following output contract. '
                   'Return actual observed values, not the schema itself. Do not fill optional '
                   'array capacity with repeated or guessed entries.\n'
                   + json.dumps(prepared['schema'], ensure_ascii=False, separators=(',', ':')))
    for message in reversed(result):
        if message['role'] == 'user':
            content = message['content']
            message['content'] = (content + instruction if isinstance(content, str)
                                  else [*content, {'type': 'text', 'text': instruction}])
            prepared['receipt']['schema_visible_to_model'] = True
            prepared['receipt']['schema_prompt_characters'] = len(instruction)
            return result
    raise ValueError('Structured output requires a user message for the visible output contract')


def infer(body):
    global weight_verified,weights_receipt,loaded,inference_phase,gpu_release_status,retained_gpu_lease,last_gpu_cleanup
    if body.model!=MODEL_ID or body.stream:raise ValueError('Fixed local model and nonstreaming requests only')
    seed=request_controls(body)
    prepared_format=prepare_response_format(body)
    messages=None;images=[];image_sources=[];image_records=[]
    try:
        messages,images,image_sources,image_records=prepare(body.messages)
        messages=schema_conditioned_messages(messages,prepared_format)
        if retained_gpu_lease is not None or gpu_release_status=='unconfirmed':
            raise HTTPException(503,'Previous GPU cleanup is unconfirmed; restart this local profile before another inference')
        import torch
        from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU or cloud fallback')
        if not lock.acquire(blocking=False):raise HTTPException(409,'Local model is busy; no duplicate inference started')
    except BaseException:
        # ``prepare`` already cleans a partial result.  This also covers every
        # post-prepare rejection before the GPU-owned try/finally below.
        close_prepared_images(images,image_sources)
        raise
    model=processor=inputs=outputs=generated=None;lease=None;started=time.monotonic()
    generation_kwargs=prose_kwargs=repeat_guard=None
    failure=None;response=None;baseline={'state':'not_initialized','allocated_bytes':0,'reserved_bytes':0}
    inference_phase='preparing';gpu_release_status='in_use'
    try:
        lease=acquire_gpu_lease()
        if not weight_verified:
            weights_receipt=verify_model_weights()
            weight_verified=True
        torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        baseline=cuda_memory_snapshot()
        # Do not override the checkpoint's saved image processor with fixed
        # service values.  It controls native visual resizing/tokenization;
        # the resulting *combined* prompt is then checked against actual
        # model context and VRAM below without dropping any image.
        processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
        image_processor=getattr(processor,'image_processor',None)
        if image_processor is None:
            raise ValueError('Fixed local checkpoint has no image processor for native image budgeting')
        image_memory_receipt=preflight_images_for_processor(images,image_records,image_processor)
        # Orientation is a real decode/transform allocation, so it follows the
        # dynamic RAM admission above and precedes Qwen's native processor.
        apply_exif_orientation(images,messages,image_records)
        template_kwargs,thinking_receipt=template_controls(body,processor)
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**template_kwargs)
        inputs=processor(text=[prompt],images=images or None,padding=True,return_tensors='pt')
        prompt_tokens=int(inputs['input_ids'].shape[-1])
        output_tokens,memory_receipt=output_token_budget(torch,prompt_tokens,body.max_tokens)
        grids=inputs.get('image_grid_thw')
        image_grids=grids.tolist() if grids is not None else []
        native_image_budget={key:getattr(image_processor,key,None) for key in ('min_pixels','max_pixels')}
        inference_phase='loading_model'
        model=Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL),local_files_only=True,
              dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
        loaded=True;loaded_seconds=time.monotonic()-started
        attention_receipt=configure_model_attention(model)
        inputs=inputs.to('cuda:0')
        stop_kwargs,stops=generation_stop_controls(body,processor)
        sampling_kwargs=generation_sampling_controls(body,model)
        guided_kwargs,guided_stats=guided_generation_controls(prepared_format,processor,model,prompt_tokens)
        generation_kwargs={**sampling_kwargs,**stop_kwargs,**guided_kwargs}
        prose_kwargs,prose_receipt,repeat_guard=generation_prose_controls(body,model,prompt_tokens,
            existing_stopping=generation_kwargs.get('stopping_criteria'))
        generation_kwargs.update(prose_kwargs)
        inference_phase='generating'
        with torch.inference_mode():
            # Do not impose a service wall-clock cutoff. EOS, explicit
            # semantic stops, repetition protection, and the model's actual
            # context/VRAM capacity are the only generation boundaries.
            outputs=model.generate(**inputs,max_new_tokens=output_tokens,**generation_kwargs)
        generated=outputs[:,prompt_tokens:]
        result=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        raw_result=processor.batch_decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)[0] if stops else result
        result,matched_stop=trim_stop_output(result,raw_result,stops)
        eos=model.generation_config.eos_token_id
        eos_ids=set(eos if isinstance(eos,list) else [eos])
        ended=bool(generated.shape[-1] and int(generated[0,-1]) in eos_ids)
        finish=generation_finish_reason(ended,matched_stop,int(generated.shape[-1]),output_tokens,repeat_guard)
        if repeat_guard is not None:prose_receipt.update(guard=repeat_guard.receipt())
        format_receipt=validate_guided_output(result,prepared_format,guided_stats,ended,finish,int(generated.shape[-1]))
        response={'id':'localchat_'+uuid.uuid4().hex,'object':'chat.completion','model':MODEL_ID,
          'choices':[{'index':0,'message':{'role':'assistant','content':result},'finish_reason':finish}],
          'usage':{'prompt_tokens':prompt_tokens,'completion_tokens':int(generated.shape[-1]),'total_tokens':int(outputs.shape[-1])},
          'factory_receipt':{'model_id':MODEL_ID,'model_directory':str(MODEL),'input_image_count':len(images),
            'profile':PROFILE,'weights_verification':weights_receipt,'source_components':SOURCE_COMPONENTS,
             'image_sha256':[record['sha256'] for record in image_records],
             'source_images':image_records,
            'image_grid_thw':image_grids,'native_image_processor_budget':native_image_budget,
             'image_memory_preflight':image_memory_receipt,
            'manual_image_edge_cap':None,'prepared_image_dimensions':[[image.width,image.height] for image in images],
            'max_output_tokens':output_tokens,'generation_time_limit_seconds':None,
            'generation_stop_policy':'model_eos_semantic_stop_or_physical_context_vram','finish_reason':finish,
            'seed':seed,'seed_source':'request' if body.seed is not None else 'fixed_default',
            'chat_template_control':thinking_receipt,'stop_sequences':stops,'matched_stop':matched_stop,
            'decoding':'sampling' if sampling_kwargs['do_sample'] else 'greedy',
            'sampling_parameters':sampling_kwargs,'requested_temperature':body.temperature,
            'prose_repetition_control':prose_receipt,
            'context_budget':{'max_model_len':model_context_limit(),'max_prompt_tokens':None,
                              'actual_prompt_tokens':prompt_tokens,'requested_output_tokens':body.max_tokens,
                              'effective_output_tokens':output_tokens},
            'vram_preflight':memory_receipt,'gpu_slot':'exclusive_local_qwen','attention_control':attention_receipt,
            'response_format_control':format_receipt,
            'device':torch.cuda.get_device_name(),'torch':torch.__version__,'offline':True,
            'load_seconds':round(loaded_seconds,3),'elapsed_seconds':round(time.monotonic()-started,3),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'script_sha256':SCRIPT_SHA}}
    except Exception as error:
        failure=exception_http_details(error)
        discard_exception_frames(error)
    finally:
        # Leave the original except block first: suppress_context alone would
        # still retain error.__context__ and its model.generate frame/weights.
        inference_phase='cleaning_gpu'
        for image in images or ():
            try:image.close()
            except Exception:pass
        for source in image_sources or ():
            try:source.close()
            except Exception:pass
        model=processor=inputs=outputs=generated=messages=images=image_sources=None
        generation_kwargs=prose_kwargs=repeat_guard=None
        last_gpu_cleanup=cleanup_gpu(torch,baseline)
        loaded=not last_gpu_cleanup['released']
        gpu_release_status='confirmed' if last_gpu_cleanup['released'] else 'unconfirmed'
        inference_phase='idle' if last_gpu_cleanup['released'] else 'gpu_release_unconfirmed'
        try:
            if last_gpu_cleanup['released']:
                release_gpu_lease(lease)
            else:
                retained_gpu_lease=lease  # Keep other local profiles from loading.
        finally:lock.release()
    if failure is not None:
        status,detail,headers=failure
        if isinstance(detail,dict):detail={**detail,'gpu_cleanup':last_gpu_cleanup,
                                           'source_images':image_records,'image_count':len(image_records)}
        elif not last_gpu_cleanup['released']:detail=str(detail)+'; GPU release unconfirmed, local profile restart required'
        raise HTTPException(status,detail,headers=headers) from None
    if not last_gpu_cleanup['released']:
        raise HTTPException(503,{'code':'gpu_release_unconfirmed','gpu_cleanup':last_gpu_cleanup}) from None
    response['factory_receipt']['gpu_cleanup']=last_gpu_cleanup
    return response

@app.post('/v1/chat/completions')
def chat(body:ChatRequest):
    import psutil
    trace_id = body.trace_id or uuid.uuid4().hex
    sources = []
    process = psutil.Process()
    stop = threading.Event()
    cpu_start = process.cpu_times()
    resources = {'peak_rss_bytes': process.memory_info().rss, 'sampling_interval_seconds': 0.1}

    def observe():
        while not stop.wait(0.1):
            resources['peak_rss_bytes'] = max(resources['peak_rss_bytes'], process.memory_info().rss)

    monitor = threading.Thread(target=observe, daemon=True)
    monitor.start()
    try:
        response = infer(body)
        sources = response['factory_receipt'].get('source_images', [])
        response['factory_receipt'].update(trace_id=trace_id, source_images=sources,
            image_count=len(sources), resource_usage=resources,
            preprocessing=('header-only inspection, dynamic host/cgroup RAM preflight, '
                           'checkpoint-native JPEG draft when saved processor metadata permits, '
                           'native Qwen processor resize/token budgeting'))
        return response
    except Exception as error:
        failure=exception_http_details(error)
        discard_exception_frames(error)
    finally:
        stop.set()
        monitor.join(timeout=1)
        cpu_end = process.cpu_times()
        resources['cpu_seconds'] = cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system
    # Raise outside except so the HTTP error cannot retain the original model.
    status,detail,headers=failure
    if isinstance(detail, dict):
        sources=detail.get('source_images',sources)
        detail = {**detail, 'trace_id': trace_id, 'stage': 'local_qwen_chat', 'image_count': len(sources),
                  'source_images': sources, 'resource_usage': resources}
    raise HTTPException(status,detail,headers=headers) from None

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8771,workers=1)
