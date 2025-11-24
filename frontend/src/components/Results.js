import React, { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

function Results() {
  const [results, setResults] = useState([]);

  const fetchResults = async () => {
    try {
      const res = await fetch("http://localhost:8000/results");
      const data = await res.json();
      setResults(data.results);
    } catch (error) {
      console.error(error);
      alert("Failed to fetch results.");
    }
  };

  useEffect(() => {
    fetchResults();
  }, []);

  return (
    <div className="results-container">
      <h3>Past Analysis Results</h3>

      {results.length === 0 ? (
        <p>No results yet.</p>
      ) : (
        <>
          {/* 📋 Table View */}
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>File Name</th>
                <th>Detected Persons</th>
                <th>Created At</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td>{r.filename}</td>
                  <td>{r.person_count}</td>
                  <td>{r.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* 📊 Chart View */}
          <div style={{ width: "100%", height: 400, marginTop: 40 }}>
            <h4>Detected Persons Over Time</h4>
            <ResponsiveContainer>
              <LineChart data={results.slice().reverse()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="created_at" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="person_count" stroke="#2980b9" activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}

export default Results;