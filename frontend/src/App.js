import { useEffect, useState, Suspense } from "react";

const CLASS_NAMES = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck"
];

function App() {
  const [categories, setCategories] = useState({});
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [modelType, setModelType] = useState("cnn");
  const [visibleCategory, setVisibleCategory] = useState(null);
  const [uploadFile, setUploadFile] = useState(null);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/categories")
      .then((res) => res.json())
      .then((data) => {
        const filteredData = {};
        CLASS_NAMES.forEach((className) => {
          if (data[className]) {
            filteredData[className] = data[className].slice(0, 5); // Tải trước 5 ảnh
          }
        });
        setCategories(filteredData);
      })
      .catch((err) => console.error("Error fetching categories:", err));
  }, []);

  const handlePredict = () => {
    if (!selectedImage) return;
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_url: selectedImage, model_type: modelType }),
    })
      .then((res) => res.json())
      .then((data) => setPrediction(data))
      .catch((err) => console.error("Error predicting:", err));
  };
  

  const handleUpload = (e) => {
    e.preventDefault();
    if (!uploadFile) return;
    
    // Tạo preview URL cho file vừa upload
    const previewUrl = URL.createObjectURL(uploadFile);
    setSelectedImage(previewUrl);
  
    // Chuẩn bị gửi file lên server để dự đoán
    const formData = new FormData();
    formData.append("file", uploadFile);
    formData.append("model_type", modelType);
  
    fetch("http://127.0.0.1:5000/upload", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => setPrediction(data))
      .catch((err) => console.error("Error uploading image:", err));
  };
  

  return (
    <div>
      <h1>CIFAR-10 Image Classifier</h1>

      <label>Select Model:</label>
      <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
        <option value="cnn">CNN</option>
        <option value="resnet">ResNet</option>
        <option value="vgg">VGG</option>
      </select>

      {/* Phần upload ảnh */}
      <div style={{ marginTop: 20 }}>
        <h2>Upload Image for Prediction</h2>
        <form onSubmit={handleUpload}>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setUploadFile(e.target.files[0])}
          />
          <button type="submit">Upload & Predict</button>
        </form>
      </div>

      {/* Phần chọn ảnh từ danh mục */}
      {Object.keys(categories).length === 0 ? (
        <p>Loading images...</p>
      ) : (
        <Suspense fallback={<p>Loading images...</p>}>
          {CLASS_NAMES.map((category) => (
            <div key={category}>
              {/* Click vào tên folder để toggle hiển thị ảnh */}
              <h2
                onClick={() =>
                  setVisibleCategory(visibleCategory === category ? null : category)
                }
                style={{ cursor: "pointer", color: "blue" }}
              >
                {category}
              </h2>
              {visibleCategory === category &&
                categories[category]?.map((img) => {
                  const imgUrl = `http://127.0.0.1:5000/image/${category}/${img}`;
                  return (
                    <img
                      key={img}
                      src={imgUrl}
                      alt={img}
                      width={100}
                      loading="lazy"
                      style={{ cursor: "pointer", margin: 5 }}
                      onClick={() => setSelectedImage(imgUrl)}
                    />
                  );
                })}
            </div>
          ))}
        </Suspense>
      )}

      {selectedImage && (
        <div>
          <h2>Selected Image</h2>
          <img src={selectedImage} alt="Selected" width={200} />
          <button onClick={handlePredict}>Predict</button>
        </div>
      )}

      {prediction && (
        <div>
          <h2>Prediction Result</h2>
          {prediction.error ? (
            <p>Error: {prediction.error}</p>
          ) : (
            <>
              <p>Class: <strong>{prediction.class_name}</strong></p>
              <p>Confidence: <strong>{(prediction.confidence * 100).toFixed(2)}%</strong></p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
