import cv2
import torchvision.transforms as transforms
import torch
from filter import HLS_filter
from PIL import Image
import numpy as np

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

LOAD_FILE = "model_950v2.pth"
model = torch.load(LOAD_FILE)
model.eval()
SCALE = [180, 255, 255, 180, 255, 255]
FILE_VIDEO_BRIGHT = "video/12.mp4"
FILE_VIDEO_DARK = "video/video1.mp4"
FILE_VIDEO_NORMAL = "video/13.mp4"
cap = cv2.VideoCapture(FILE_VIDEO_NORMAL)

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer= cv2.VideoWriter('/home/ntnuerc/ilham/Machine_Learning/FINAL_EXAM_CNN+NN/result filtered/Dark.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640,480))

def video_test():
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame,(256,256))
        
        if not ret:
            break

        test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        test_image_pil = Image.fromarray(test_image)  # Convert NumPy array to PIL image
        preprocessed_image = preprocess(test_image_pil)
        input_tensor = preprocessed_image.unsqueeze(0)  # Add batch dimension

        # If available, move the input tensor to the GPU for faster inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        model.to(device)

        # Perform the forward pass
        with torch.no_grad():
            output = model(input_tensor)
            hls_values = output.squeeze().cpu().numpy() * SCALE
            # formatted_arr = np.array2string(hls_values, precision=3, suppress_small=True)
            # print('HLS value:', formatted_arr)
        # Apply HLS filtering to the test image
        filtered = HLS_filter(frame, hls_values)
        noise_reduce = cv2.morphologyEx(filtered,cv2.MORPH_CLOSE,kernel=(5,5))
        binary_video = cv2.cvtColor(filtered,cv2.COLOR_GRAY2BGR)
        writer.write(binary_video)
        # print(filtered.shape)
        # Display the original frame and the filtered frame
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Filtered Frame', filtered)
        cv2.imshow('Opening',noise_reduce)

        
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

# Call the video_test function to start processing the video
video_test()
