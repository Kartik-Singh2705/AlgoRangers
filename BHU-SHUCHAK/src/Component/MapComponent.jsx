import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useContext, useEffect } from 'react';
import { AppContext } from './AppContext';

// Helper to update view when coordinates change
function ChangeView({ coords }) {
  const map = useMap();
  useEffect(() => {
    if (coords) {
      map.flyTo(coords, map.getZoom()); // smooth animation
    }
  }, [coords, map]);
  return null;
}

export default function MapComponent() {
  const { coordinates, location } = useContext(AppContext);

  return (
    <MapContainer center={coordinates} zoom={7} style={{ height: "500px", width: "100%" }}>
    
      <TileLayer
        url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
        attribution="Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)"
      />

      <Marker position={coordinates}>
        <Popup>⛰️ {location}</Popup>
      </Marker>

      <ChangeView coords={coordinates} />
    </MapContainer>
  );
}
