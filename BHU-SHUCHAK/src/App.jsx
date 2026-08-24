import React from 'react'
import NavBar from './Component/NavBar'
import Landslide from './Component/Landslide'
import Charts from './Component/Charts'
import LiveCameraFeed from './Component/LiveCameraFeed'

const App = () => {
  return (
    <>
      <NavBar />
      <Landslide></Landslide>
      <Charts></Charts>
      <LiveCameraFeed></LiveCameraFeed>
    </>
  )
}

export default App