import sys
sys.path.append('./')
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from flask import (flash, Flask, redirect, render_template, request,
                   session, url_for, send_file, jsonify, send_from_directory)
from flask_cors import CORS
import uuid
import concurrent.futures

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# create a flask app instance
app = Flask(__name__)

# Apply cors policy in our app instance
CORS(app)

# setup all config variable
app.config["enviroment"] = "prod"
app.config["SECRET_KEY"] = uuid.uuid4().hex

# handling our application secure type like http or https
secure_type = "http"

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# allow only that image file extension
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg', 'webp'}

def allowed_photos(filename):
    """
    checking file extension is correct or not

    :param filename: file name
    :return: True, False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/view_logs", methods=['GET'])
def view_logs():
    try:
        server_file_name = "server.log"
        file = os.path.abspath(server_file_name)
        lines = []
        with open(file, "r") as f:
            lines += f.readlines()
        response = {"status_code": 200, "data": lines}
        return response

    except Exception as e:
        return {"message": "data is not present"}

def pil_to_binary_mask(pil_image, threshold=0):
    print("new1")
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    print("new2")
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    print("new3")
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')


tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    )
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

def start_tryon(dict,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed, folder_image_store_path, category):
    try:
        openpose_model.preprocessor.body_estimation.model.to(device)
        pipe.to(device)
        pipe.unet_encoder.to(device)
        print("task1")
        garm_img= garm_img.convert("RGB").resize((768,1024))
        human_img_orig = dict["background"].convert("RGB")    
        
        if is_checked_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768,1024))
        else:
            human_img = human_img_orig.resize((768,1024))
        print("task2")

        print(is_checked)
        if is_checked:
            print("coming in here")
            print(human_img)
            keypoints = openpose_model(human_img.resize((384,512)))
            print("bsdhjbajdhb")
            model_parse, _ = parsing_model(human_img.resize((384,512)))
            print("newdata")
            mask, mask_gray = get_mask_location('hd', category)
            mask = mask.resize((768,1024))
        else:
            mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
            # mask = transforms.ToTensor()(mask)
            # mask = mask.unsqueeze(0)
        mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray+1.0)/2.0)
        print("task3")


        human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        print("task4")
        
        
        yaml_file_path = os.path.abspath('configs/densepose_rcnn_R_50_FPN_s1x.yaml')
        ckpt_file_path = os.path.abspath('ckpt/densepose/model_final_162be9.pkl')
        args = apply_net.create_argument_parser().parse_args(('show', yaml_file_path, ckpt_file_path, 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
        # verbosity = getattr(args, "verbosity", None)
        pose_img = args.func(args,human_img_arg)    
        pose_img = pose_img[:,:,::-1]    
        pose_img = Image.fromarray(pose_img).resize((768,1024))
        print("task5")
        
        with torch.no_grad():
            # Extract the images
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    prompt = "model is wearing " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    print("task6")
                    
                    with torch.inference_mode():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )
                        print("task7")
                                 
                        prompt = "a photo of " + garment_des
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                        if not isinstance(prompt, List):
                            prompt = [prompt] * 1
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * 1
                        with torch.inference_mode():
                            (
                                prompt_embeds_c,
                                _,
                                _,
                                _,
                            ) = pipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                negative_prompt=negative_prompt,
                            )


                        print("task8")

                        pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                        garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device,torch.float16),
                            negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength = 1.0,
                            pose_img = pose_img.to(device,torch.float16),
                            text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                            cloth = garm_tensor.to(device,torch.float16),
                            mask_image=mask,
                            image=human_img, 
                            height=1024,
                            width=768,
                            ip_adapter_image = garm_img.resize((768,1024)),
                            guidance_scale=2.0,
                        )[0]
        print("image generate success. ongoing storing")
        if is_checked_crop:
            out_img = images[0].resize(crop_size)        
            human_img_orig.paste(out_img, (int(left), int(top)))
            try:
                human_img_orig.save(folder_image_store_path)
            except:
                human_img_orig.save(folder_image_store_path)
            return human_img_orig, mask_gray
        else:
            human_img_orig = images[0]
            try:
                human_img_orig.save(folder_image_store_path)
            except:
                human_img_orig.save(folder_image_store_path)
            return images[0], mask_gray
        
    except Exception as e:
        print(f"Error in function: {e}")

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

##default human


# image_blocks = gr.Blocks().queue()
# with image_blocks as demo:
#     gr.Markdown("## IDM-VTON 👕👔👚")
#     gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
#     with gr.Row():
#         with gr.Column():
#             imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)
#             with gr.Row():
#                 is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)",value=True)
#             with gr.Row():
#                 is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing",value=False)
#
#             example = gr.Examples(
#                 inputs=imgs,
#                 examples_per_page=10,
#                 examples=human_ex_list
#             )
#
#         with gr.Column():
#             garm_img = gr.Image(label="Garment", sources='upload', type="pil")
#             with gr.Row(elem_id="prompt-container"):
#                 with gr.Row():
#                     prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
#             example = gr.Examples(
#                 inputs=garm_img,
#                 examples_per_page=8,
#                 examples=garm_list_path)
#         with gr.Column():
#             # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
#             masked_img = gr.Image(label="Masked image output", elem_id="masked-img",show_share_button=False)
#         with gr.Column():
#             # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
#             image_out = gr.Image(label="Output", elem_id="output-img",show_share_button=False)
#
#
#
#
#     with gr.Column():
#         try_button = gr.Button(value="Try-on")
#         with gr.Accordion(label="Advanced Settings", open=False):
#             with gr.Row():
#                 denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
#                 seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)
#
#
#
#     try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, is_checked,is_checked_crop, denoise_steps, seed], outputs=[image_out,masked_img], api_name='tryon')
# from PIL import Image
# human_image_path = "gradio_demo/try/model_image.webp"
# garment_image_path = "gradio_demo/try/garment.webp"
# human_image_path = os.path.abspath(human_image_path)
# garment_image_path = os.path.abspath(garment_image_path)

@app.route("/stylic/take-photo", methods=["GET", "POST"])
def take_photo():
    """
    In this route we can handling superadmin data
    :return: superadmin template
    """
    try:
        print("coming in take photo")
        files_uploaded = []
        folder_person_name = request.form.get("folder_name")
        category = request.form.get("category")
        print(category)
        folder_image_store_path = f"static/uploads/{folder_person_name}"
        os.makedirs(folder_image_store_path, exist_ok=True)
        file1 = request.files.get("garment_file")
        if file1 and file1.filename != "":
            exten = file1.filename.split(".")[-1]
            file1_path = os.path.join(folder_image_store_path, f"garment.{exten}")
            file1.save(file1_path)
            files_uploaded.append(file1_path.replace("\\", "/"))
        print("upload garment successfully")
        # Handle second file
        file2 = request.files.get("model_file")
        if file2 and file2.filename != "":
            exten1 = file2.filename.split(".")[-1]
            file2_path = os.path.join(folder_image_store_path, f"model.{exten1}")
            file2.save(file2_path)
            files_uploaded.append(file2_path.replace("\\", "/"))
        print("upload model successfully")

        if not files_uploaded:
            return "No files selected for upload"
        
        from PIL import Image
        human_image = Image.open(files_uploaded[1])
        garment_image = Image.open(files_uploaded[0])
        # start_tryon({"background": human_image}, garment_image, "", True, False, 30, 42)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
        threads = []
        print("generate photoshoot")
        output_folder_image_store_path = os.path.join(folder_image_store_path, "output.jpg")
        start_tryon({"background": human_image}, garment_image, "", True, False, 30, 42, output_folder_image_store_path, category)
        # concurrent.futures.wait(threads)
        print("generated_successfully")
        response = {"status_code": 200, "data": {"output_file": f"http://139.84.138.54:80/download_photo/{folder_image_store_path.replace('/', '---')}***output.jpg"}}
        return response

    except Exception as e:
        return {"message": "data is not present"}

@app.route("/stylic/take-photoshoot", methods=["GET", "POST"])
def photoshoot():
    """
    In this route we can handling superadmin data
    :return: superadmin template
    """
    try:
        files_uploaded = []
        folder_person_name = request.form.get("folder_name")
        folder_image_store_path = f"static/uploads/{folder_person_name}"
        os.makedirs(folder_image_store_path, exist_ok=True)
        print("request are coming")
        file1 = request.files.get("garment_file")
        if file1 and file1.filename != "":
            exten = file1.filename.split(".")[-1]
            file1_path = os.path.join(folder_image_store_path, f"garment.{exten}")
            file1.save(file1_path)
            files_uploaded.append(file1_path.replace("\\", "/"))
        print("garment file uploaded")
        # Handle second file
        file2 = request.files.get("model_file")
        if file2 and file2.filename != "":
            exten1 = file2.filename.split(".")[-1]
            file2_path = os.path.join(folder_image_store_path, f"model.{exten1}")
            file2.save(file2_path)
            files_uploaded.append(file2_path.replace("\\", "/"))
        print("model1 file uploaded")  
        # Handle second file
        file3 = request.files.get("model_file1")
        if file3 and file3.filename != "":
            exten3 = file3.filename.split(".")[-1]
            file3_path = os.path.join(folder_image_store_path, f"model1.{exten3}")
            file3.save(file3_path)
            files_uploaded.append(file3_path.replace("\\", "/"))
        print("model2 file uploaded") 
        # Handle second file
        file4 = request.files.get("model_file2")
        if file4 and file4.filename != "":
            exten4 = file4.filename.split(".")[-1]
            file4_path = os.path.join(folder_image_store_path, f"model.{exten4}")
            file4.save(file4_path)
            files_uploaded.append(file4_path.replace("\\", "/"))
        print("model3 file uploaded")
        if not files_uploaded:
            return "No files selected for upload"
        
        from PIL import Image
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
        threads = []
        index_list = [1,2,3]
        all_output_list = []
        for index in index_list:
            human_image = Image.open(files_uploaded[index])
            garment_image = Image.open(files_uploaded[0])
            print("generate photoshoot")
            output_folder_image_store_path = os.path.join(folder_image_store_path, f"output{index}.jpg")
            start_tryon({"background": human_image}, garment_image, "", True, False, 30, 42, output_folder_image_store_path)
            all_output_list.append(f"http://139.84.138.54:80/download_photo/{folder_image_store_path.replace('/', '---')}***output{index}.jpg")
        # concurrent.futures.wait(threads)
        print("generated_successfully")
        response = {"status_code": 200, "data": {"output_file": all_output_list}}
        return response

    except Exception as e:
        return {"message": "data is not present"}

@app.route("/download_photo/<folder_path_image>", methods=["GET"])
def folder_store_path(folder_path_image):
    """
    In this route we can handling superadmin data
    :return: superadmin template
    """
    try:
        print(folder_path_image)
        all_list = folder_path_image.split("***")
        folder_path = all_list[0].replace("---", "/")
        filename = all_list[1]
        print(folder_path, filename)
        return send_from_directory(folder_path, filename, as_attachment=True)

    except Exception as e:
        return {"message": "data is not present"}


# Load images
# human_image = Image.open(human_image_path)
# garment_image = Image.open(garment_image_path)
# start_tryon({"background": human_image}, garment_image, "", True, False, 30, 42)


# image_blocks.launch()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)