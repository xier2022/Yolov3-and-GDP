// app.js

// When OpenCV.js is ready
cv['onRuntimeInitialized'] = () => {
    // Get video element
    const video = document.getElementById('videoElement');

    // Get canvas element
    const canvas = document.getElementById('canvasOutput');
    const context = canvas.getContext('2d');

    // Access user media
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            // Set video source to the stream
            video.srcObject = stream;

            // Load face detection classifier
            const classifier = new cv.CascadeClassifier();
            classifier.load('haarcascade_frontalface_default.xml');

            // Start video processing loop
            setInterval(() => {
                // Capture current frame from video
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

                // Convert image data to grayscale
                const src = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC4);
                cv.cvtColor(cv.matFromImageData(imageData), src, cv.COLOR_RGBA2GRAY);

                // Detect faces
                const faces = new cv.RectVector();
                classifier.detectMultiScale(src, faces, 1.1, 3, 0);

                // Draw rectangles around detected faces
                for (let i = 0; i < faces.size(); ++i) {
                    const face = faces.get(i);
                    const point1 = new cv.Point(face.x, face.y);
                    const point2 = new cv.Point(face.x + face.width, face.y + face.height);
                    cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
                }

                // Show processed image on canvas
                cv.imshow('canvasOutput', src);

                // Release resources
                src.delete();
                faces.delete();
            }, 100);
        })
        .catch((err) => console.error('Error accessing the camera:', err));
};
