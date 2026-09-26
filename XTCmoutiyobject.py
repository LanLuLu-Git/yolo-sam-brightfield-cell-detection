import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Ensure Chinese characters render correctly
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def show_anns(anns):
    """Visualize segmentation masks."""
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # Sort by area from largest to smallest
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    for ann in sorted_anns:
        m = ann['segmentation']
        # Randomly generate a mask color
        color_mask = np.random.random(3)
        # Convert the mask to color and overlay it
        img = np.ones((m.shape[0], m.shape[1], 3))
        img[:, :, 0] = color_mask[0]
        img[:, :, 1] = color_mask[1]
        img[:, :, 2] = color_mask[2]
        ax.imshow(np.dstack((img, m * 0.4)))


def segment_multi_objects(image_path, model_type='vit_h', checkpoint_path='D:/Project/YOLOwithSAM/segment-anything-main/models/sam_vit_h_4b8939.pth', output_path=None):
    """
    Segment multiple objects in an image using the SAM model.

    Args:
        image_path: input image path
        model_type: model type ('vit_h', 'vit_l', 'vit_b')
        checkpoint_path: model checkpoint path
        output_path: output image save path; if None, display the image instead
    """
    # Load the model
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    # Create the mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=15,  # number of sampling points
        pred_iou_thresh=0.90,  # IoU threshold
        stability_score_thresh=0.95,  # stability score threshold
        crop_n_layers=1,  # number of crop layers; controls whether/how many times the image is cropped when generating masks (1 = single crop)
        crop_n_points_downscale_factor=2,  # downscale factor for sampling points when cropping
        # controls the number of points sampled on the cropped image relative to the original
        # 2 means half the number of points of the original image
        min_mask_region_area=40000,  # filter small regions
    )

    # Load the image
    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    print("Segmenting objects in the image...")
    masks = mask_generator.generate(image)
    print(f"Detected {len(masks)} objects")

    # Visualize the result
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title("Original image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(122)
    plt.title(f"Segmented {len(masks)} objects")
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')

    plt.tight_layout()

    # Save or display the result
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Result saved to: {output_path}")
    else:
        plt.show()

    return masks


if __name__ == "__main__":
    # Modify the image path here as needed
    input_image = "D:/Project/YOLOwithSAM/ultralytics-main/ultralytics/assets/bus.jpg"

    # Optional: specify an output image path by uncommenting the line below
    # output_image = "segmented_result.jpg"

    try:
        masks = segment_multi_objects(
            image_path=input_image,
            model_type='vit_h',  # options: 'vit_l', 'vit_b'
            checkpoint_path='D:/Project/YOLOwithSAM/segment-anything-main/models/sam_vit_h_4b8939.pth'
            # output_path=output_image  # uncomment to save the result
        )
    except Exception as e:
        print(f"Error during processing: {e}")
