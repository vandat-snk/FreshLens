# Fruit Grad-CAM demo provenance

Input: `fruit_input_ai_demo.png`, generated with the built-in image_gen tool. This is a synthetic illustration, not a dataset sample or a real camera photograph.

Prompt: Create a square photorealistic raw input photo for a fruit classification demo: one single fresh red apple, fully visible with short brown stem, centered occupying 70 percent of frame on a plain off-white tabletop and neutral pale background. Natural soft daylight, realistic skin texture, sharp focus, everyday camera photograph. No text, no labels, no collage, no heatmap, no interface. Save output as an image and return local path if available.

Output: `gradcam_fruit_ai_demo.png`, computed by `GENERATE_GRADCAM_DEMO.py` using the local `models/cnn_efficientnet_b0/best.pt` checkpoint. Predicted class: `apple::fresh`. See the adjacent JSON for checkpoint and image hashes.

These demo assets are displayed as a fixed, explicitly labeled synthetic example on the app Explanation page (ui/explanation.py).
