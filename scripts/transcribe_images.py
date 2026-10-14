"""Recognize cropped manuscript line images with a supplied trained TrOCR checkpoint."""
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True,help='Complete local TrOCR checkpoint, including weights and processor files')
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/htr_inference.json')
    p.add_argument('--images',type=Path,nargs='+',required=True,help='Cropped line images, not whole letter pages')
    p.add_argument('--output',type=Path,required=True,help='New JSONL output file')
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists; choose a new path')
    if not a.model_dir.is_dir():p.error('Model directory does not exist')
    for image in a.images:
        if not image.is_file():p.error('Missing image: '+str(image))
    cfg=json.loads(a.config.read_text());params={k:cfg[k] for k in ['num_beams','max_new_tokens','do_sample','use_cache']}
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor,VisionEncoderDecoderModel
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    processor=TrOCRProcessor.from_pretrained(str(a.model_dir),local_files_only=True,use_fast=False)
    model=VisionEncoderDecoderModel.from_pretrained(str(a.model_dir),local_files_only=True).to(device).eval()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as out,torch.inference_mode():
        for i,path in enumerate(a.images,1):
            with Image.open(path) as im: pixels=processor(images=im.convert('RGB'),return_tensors='pt').pixel_values.to(device)
            generated=model.generate(pixels,**params)
            text=processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
            row={'line_id':str(path.resolve()),'image':str(path.resolve()),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'text':text,'text_source':'htr_output','model_directory':str(a.model_dir.resolve()),'generation':params,'verification_status':'Unverified HTR output'}
            out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(f'{i}/{len(a.images)}: {text}',flush=True)

if __name__=='__main__':main()
