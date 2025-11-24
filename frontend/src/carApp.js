import { useState } from "react";
import axios from "axios";

function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    setPreview(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post("http://localhost:8000/upload-plate/", formData);
    setResult(res.data);
  };

  return (
    <div style={{ padding: 50 }}>
      <h1>차량 번호판 인식</h1>

      <input type="file" onChange={handleUpload} />

      {preview && (
        <>
          <h2>❗ 업로드된 이미지</h2>
          <img src={preview} width="300" />
        </>
      )}

      {result && (
        <>
          <h2>📌 결과</h2>
          <p>번호판 텍스트 : {result.plate_text}</p>
          <img src={`data:image/jpg;base64,${result.plate_image}`} width="250" />
        </>
      )}
    </div>
  );
}

export default App;