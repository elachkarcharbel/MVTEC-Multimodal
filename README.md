# MVTEC-AD-Caption
Generating a Dual-modality of the MVTEC-AD dataset by leveraging LLMs for Image Captioning generation

Below is the list of pretrained LLM used to generate captions out of the MVTEC-AD dataset images.

Model Name | Link to full code implementation |  
--- | :---:
BLIP |  <a href="https://github.com/salesforce/BLIP">View Code</a> |
CLIP(Flickr30k)+GPT2 |  <a href="https://github.com/jmisilo/clip-gpt-captioning">View Code</a> |
CLIP(COCO+Conceptual)+GPT2 | <a href="https://github.com/rmokady/CLIP_prefix_caption">View Code</a> |


<table>
  <tr>
    <th>Pretrained Model</th>
    <th>Link to download the models</th>
  </tr>
  <tr>
    <td>BLIP w/ ViT-B and CapFilt-L</td>
    <td><a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth">Download</a></td>
  </tr>
  <tr>
    <td>BLIP w/ ViT-L</td>
    <td><a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth">Download</a></td>
  </tr>
  <tr>
    <td>CLIP(Flickr30k)+GPT2-small</td>
    <td><a href="https://drive.google.com/file/d/1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF/view?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>CLIP(Flickr30k)+GPT2-large</td>
    <td><a href="https://drive.google.com/file/d/1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG/view?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>CLIP(ViT-B/16+COCO)+GPT2</td>
    <td rowspan="6"><a href="https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing">Download (pretrained COCO)</a></td>
  </tr>
  <tr>
    <td>CLIP(ViT-B/32+COCO)+GPT2</td>
  </tr>
    <tr>
    <td>CLIP(ViT-L/14+COCO)+GPT2</td>
  </tr>
    <tr>
    <td>CLIP(ViT-L/14@336px+COCO)+GPT2</td>
  </tr>
      <tr>
    <td>CLIP(RN50+COCO)+GPT2</td>
  </tr>
      <tr>
    <td>CLIP(RN101+COCO)+GPT2</td>
  </tr>
  
  <tr>
    <td>CLIP(ViT-B/16+Conceptual)+GPT2</td>
    <td rowspan="6"><a href="https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing">Download (pretrained conceptual)</a></td>
  </tr>
  <tr>
    <td>CLIP(ViT-B/32+Conceptual)+GPT2</td>
  </tr>
    <tr>
    <td>CLIP(ViT-L/14+Conceptual)+GPT2</td>
  </tr>
    <tr>
    <td>CLIP(ViT-L/14@336px+Conceptual)+GPT2</td>
  </tr>
      <tr>
    <td>CLIP(RN50+Conceptual)+GPT2</td>
  </tr>
      <tr>
    <td>CLIP(RN101+Conceptual)+GPT2</td>
  </tr>
</table>
