import React from 'react'
import SoilMoistureChart from './SoilMoistureChart'
import RainfallChart from './RainfallChart'

const Charts = () => {
  return (
    <div className='flex'>
      <RainfallChart></RainfallChart>
      <SoilMoistureChart></SoilMoistureChart>
    </div>
  )
}

export default Charts