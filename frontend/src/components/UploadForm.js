import React, { useState } from "react";

function UploadForm() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a video file!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Error during analysis.");
    }
    setLoading(false);
  };

  return (
    <div className="upload-container">
      <h3>Upload a Video</h3>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="video/*" onChange={handleFileChange} />
        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {result && (
        <div className="result-box">
          <h4>Analysis Result</h4>
          <p><strong>File:</strong> {result.filename}</p>
          <p><strong>Detected Persons:</strong> {result.person_count}</p>
          <p>{result.message}</p>
        </div>
      )}
    </div>
  );
}

export default UploadForm;