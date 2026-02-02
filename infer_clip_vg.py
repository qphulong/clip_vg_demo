import os
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from pathlib import Path

import utils.misc as utils
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-VG Inference', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', type=str, default='ViT-B/16')
    parser.add_argument('--checkpoint', type=str, required=True)
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
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.jpg')
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
    text_tokens = clip.tokenize([text], context_length=max_query_len).to(device)  # [1, max_query_len]
    
    # Create NestedTensor for text
    # Text mask: True for padding tokens, False for valid tokens
    text_mask = torch.zeros((text_tokens.shape[0], text_tokens.shape[1]), 
                            dtype=torch.bool, device=device)
    # Mark padding positions (assuming 0 is padding token)
    text_mask = (text_tokens == 0)
    
    text_data = NestedTensor(text_tokens, text_mask)
    
    # Forward pass
    outputs = model(img_data, text_data)
    
    # Extract predictions
    # The model outputs the boxes directly as a tensor [1, 4]
    pred_boxes = outputs.cpu()  # [1, 4]
    
    # Get the prediction (there's only one)
    best_box = pred_boxes[0]  # [4]
    best_score = 1.0  # No confidence score available from this output
    
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
    
    label = f'{text} ({score:.3f})'
    
    # Get text bounding box
    text_bbox = draw.textbbox((x1, max(0, y1 - 35)), label, font=font)
    
    # Draw text background
    draw.rectangle(text_bbox, fill='red')
    draw.text((x1, max(0, y1 - 35)), label, fill='white', font=font)
    
    image_copy.save(output_path)
    print(f"Output saved to: {output_path}")


def main(args):
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args, device)
    
    # Load and preprocess image
    print(f"\nProcessing image: {args.image}")
    print(f"Query: {args.text}")
    
    image, image_tensor, original_size = preprocess_image(args.image, args.imsize)
    
    # Run inference
    bbox_normalized, score = predict(model, image_tensor, args.text, device, args.max_query_len)
    
    # Rescale bbox to original image size
    bbox_original = rescale_bbox(bbox_normalized, original_size, args.imsize)
    
    print(f"\nResults:")
    print(f"  Confidence score: {score:.4f}")
    print(f"  Bounding box (x1, y1, x2, y2): {[int(x) for x in bbox_original]}")
    
    # Draw and save result
    draw_bbox_on_image(image, bbox_original, args.text, score, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP-VG Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)