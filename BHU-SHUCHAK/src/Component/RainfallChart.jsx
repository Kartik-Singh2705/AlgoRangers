import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const RainfallChart = () => {
  const labels = [
    "Assam",
    "Arunachal",
    "Meghalaya",
    "Manipur",
    "Mizoram",
    "Nagaland",
    "Tripura",
    "Sikkim",
  ];

  const data = {
    labels,
    datasets: [
      {
        label: "2023 Rainfall (mm)",
        data: [2200, 2800, 2500, 1800, 1900, 1700, 1600, 2400],
        borderColor: "rgba(255, 99, 132, 1)",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        tension: 0.4,
      },
      {
        label: "2024 Rainfall (mm)",
        data: [2300, 2900, 2600, 1900, 2000, 1750, 1650, 2500],
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        tension: 0.3,
      },
      {
        label: "2025 Rainfall (mm)",
        data: [2400, 3000, 2700, 2000, 2100, 1800, 1700, 2600],
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        tension: 0,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: "Northeast India Rainfall Trends (2023–2025)" },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: "Rainfall (mm)" },
      },
    },
  };

  return (
    <div className="m-10 w-150 h-100">
      <Line data={data} options={options} />
    </div>
  );
};

export default RainfallChart;
