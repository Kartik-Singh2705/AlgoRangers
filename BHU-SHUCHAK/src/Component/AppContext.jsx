import React, { createContext, useState } from 'react';


export const AppContext = createContext();


export const AppProvider = ({ children }) => {
  const [coordinates, setCoordinates] = useState([28.6139, 77.2090])
  const [location, setLocation] = useState("Location")

  return (
    <AppContext.Provider value={{coordinates,setCoordinates ,location,setLocation}}>
      {children}
    </AppContext.Provider>
  );
};
