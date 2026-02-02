import os
import json
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from pathlib import Path

import utils.misc as utils
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-VG Batch Inference', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', type=str, default='ViT-B/16')
    parser.add_argument('--checkpoint', type=str, default='./pretrained/referit/best_checkpoint.pth') # referit or unc is the best
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--imsize', default=224, type=int)
    parser.add_argument('--emb_size', default=512, type=int)
    parser.add_argument('--vl_dropout', default=0.1, type=float)
    parser.add_argument('--vl_nheads', default=8, type=int)
    parser.add_argument('--vl_hidden_dim', default=512, type=int)
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int)
    parser.add_argument('--vl_enc_layers', default=6, type=int)
    parser.add_argument('--vl_dec_layers', default=6, type=int)
    
    # Inference parameters
    parser.add_argument('--input_dir', type=str, default='test_infer_images/inputs/')
    parser.add_argument('--output_dir', type=str, default='test_infer_images/outputs/')
    parser.add_argument('--json_file', type=str, default='test_infer_images/test_infer_texts.json')
    parser.add_argument('--max_query_len', default=77, type=int)
    
    # System parameters
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--light', dest='light', default=False, action='store_true')
    
    return parser


def load_model(args, device):
    """Load the model from checkpoint"""
    print(f"Loading model from {args.checkpoint}")
    
    if args.model == "ViT-L/14" or args.model == "ViT-L/14@336px":
        args.vl_hidden_dim = 768
    
    model = build_model(args)
    model.to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, imsize=224):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    return image, image_tensor, original_size


def box_cxcywh_to_xyxy(bbox):
    """Convert bbox from center format to corner format"""
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def rescale_bbox(bbox, original_size, model_size=224):
    """Rescale bbox from model size to original image size"""
    width, height = original_size
    x1, y1, x2, y2 = bbox
    
    x1 = x1 * width
    y1 = y1 * height
    x2 = x2 * width
    y2 = y2 * height
    
    return [x1, y1, x2, y2]


@torch.no_grad()
def predict(model, image_tensor, text, device, max_query_len=77):
    """Run inference on a single image-text pair"""
    from utils.misc import NestedTensor
    import clip
    
    image_tensor = image_tensor.to(device)
    
    # Create NestedTensor for image as expected by the model
    img_mask = torch.zeros((image_tensor.shape[0], image_tensor.shape[2], image_tensor.shape[3]), 
                           dtype=torch.bool, device=device)
    img_data = NestedTensor(image_tensor, img_mask)
    
    # Tokenize text using CLIP tokenizer
    text_tokens = clip.tokenize([text], context_length=max_query_len).to(device)
    
    # Create NestedTensor for text
    text_mask = torch.zeros((text_tokens.shape[0], text_tokens.shape[1]), 
                            dtype=torch.bool, device=device)
    text_mask = (text_tokens == 0)
    
    text_data = NestedTensor(text_tokens, text_mask)
    
    # Forward pass
    outputs = model(img_data, text_data)
    
    # Extract predictions
    pred_boxes = outputs.cpu()  # [1, 4]
    
    # Get the prediction
    best_box = pred_boxes[0]  # [4]
    best_score = 1.0
    
    # Convert from cxcywh to xyxy format
    best_box_xyxy = box_cxcywh_to_xyxy(best_box.numpy())
    
    return best_box_xyxy, best_score


def draw_bbox_on_image(image, bbox, text, score, output_path):
    """Draw bounding box on image using PIL"""
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline='red', width=5)
    
    # Draw text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    label = f'{text}'
    
    # Get text bounding box
    text_bbox = draw.textbbox((x1, max(0, y1 - 35)), label, font=font)
    
    # Draw text background
    draw.rectangle(text_bbox, fill='red')
    draw.text((x1, max(0, y1 - 35)), label, fill='white', font=font)
    
    image_copy.save(output_path)
    print(f"  Saved: {output_path}")


def main(args):
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args, device)
    print()
    
    # Load JSON file with test queries
    print(f"Loading test queries from: {args.json_file}")
    with open(args.json_file, 'r') as f:
        test_data = json.load(f)
    print(f"Found {len(test_data)} images with test queries\n")
    
    # Process each image
    total_tests = 0
    for image_name, text_queries in test_data.items():
        image_path = os.path.join(args.input_dir, image_name)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"WARNING: Image not found: {image_path}")
            continue
        
        print(f"Processing: {image_name} ({len(text_queries)} queries)")
        
        # Load image once
        image, image_tensor, original_size = preprocess_image(image_path, args.imsize)
        
        # Process each text query for this image
        for idx, text_query in enumerate(text_queries, start=1):
            print(f"  Query {idx}: '{text_query}'")
            
            # Run inference
            bbox_normalized, score = predict(model, image_tensor, text_query, device, args.max_query_len)
            
            # Rescale bbox to original image size
            bbox_original = rescale_bbox(bbox_normalized, original_size, args.imsize)
            
            # Create output filename
            image_stem = Path(image_name).stem
            image_ext = Path(image_name).suffix
            output_name = f"{image_stem}_{idx}{image_ext}"
            output_path = os.path.join(args.output_dir, output_name)
            
            # Draw and save result
            draw_bbox_on_image(image, bbox_original, text_query, score, output_path)
            
            total_tests += 1
        
        print()
    
    print(f"Completed! Processed {total_tests} total tests across {len(test_data)} images.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP-VG Batch Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)