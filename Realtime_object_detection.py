from imports import *

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define the objects you want to detect
object_descriptions = ["a person", "a dog", "a cat", "a car", "a tree", "a chair","a football","a cup"]

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image for CLIP
    image_input = preprocess(pil_image).unsqueeze(0).to(device)

    # Prepare text inputs
    text_inputs = torch.cat([clip.tokenize(desc) for desc in object_descriptions]).to(device)

    # Get image and text features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Compute similarity scores
    similarity_scores = (image_features @ text_features.T).softmax(dim=-1)
    scores = similarity_scores.cpu().numpy()[0]

    # Get the most likely object
    max_score_index = np.argmax(scores)
    detected_object = object_descriptions[max_score_index]
    confidence = scores[max_score_index]

    # Display the detected object and confidence
    label = f"{detected_object} ({confidence:.2f})"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()