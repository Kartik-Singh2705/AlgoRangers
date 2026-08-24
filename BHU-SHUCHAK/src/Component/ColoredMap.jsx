import { MapContainer, TileLayer, Polygon, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export default function ColoredMap() {
  // Example coordinates for polygons (arrays of lat/lng pairs)
  const gangtokArea = [
    [27.3314, 88.6130],
    [27.3414, 88.6230],
    [27.3214, 88.6330],
  ];
  const kalimpongArea = [
    [27.0346, 88.6308],
    [27.0446, 88.6408],
    [27.0246, 88.6508],
  ];

  return (
    <MapContainer center={[27.0, 94.0]} zoom={7} style={{ height: "500px", width: "100%" }}>
      <TileLayer
        url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
        attribution="Map data © OpenStreetMap contributors, SRTM | Map style © OpenTopoMap"
      />

      {/* Polygon for Gangtok */}
      <Polygon pathOptions={{ color: 'red', fillColor: 'orange', fillOpacity: 0.5 }} positions={gangtokArea}>
        <Popup>Gangtok Risk Zone</Popup>
      </Polygon>

      {/* Polygon for Kalimpong */}
      <Polygon pathOptions={{ color: 'blue', fillColor: 'lightblue', fillOpacity: 0.5 }} positions={kalimpongArea}>
        <Popup>Kalimpong Risk Zone</Popup>
      </Polygon>
    </MapContainer>
  );
}
