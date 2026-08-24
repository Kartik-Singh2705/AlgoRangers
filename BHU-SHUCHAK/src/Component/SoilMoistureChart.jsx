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

const SoilMoistureChart = () => {
  const labels = ["2011", "2012", "2013", "2014", "2015", "2016"];

  const data = {
    labels,
    datasets: [
      {
        label: "Assam",
        data: [80, 85, 78, 82, 76, 79],
        borderColor: "rgba(75, 192, 192, 1)", // teal line
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        tension: 0.4, // smooth curve
      },
      {
        label: "Arunachal Pradesh",
        data: [95, 100, 92, 97, 90, 93],
        borderColor: "rgba(54, 162, 235, 1)", // blue line
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        tension: 0.4,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: "Soil Moisture Trends (2011–2016)" },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 160,
        title: { display: true, text: "Moisture Index" },
      },
    },
  };

  return (
    <div className="w-150 max-w-3xl m-10">
      <Line data={data} options={options} />
    </div>
  );
};

export default SoilMoistureChart;
