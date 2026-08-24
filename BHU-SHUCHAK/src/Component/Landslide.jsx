import React from 'react'
import { CloudRain } from 'lucide-react';
import img from "../assets/Landslideimg.png"
import MapComponent from './MapComponent';
import ColoredMap from './ColoredMap';

const Landslide = () => {
  return (
    <>
    <div className='flex m-5'>
    <div>
      <div className='w-70 h-20 bg-blue-950 rounded m-5 flex items-center pl-5'>
          <div className='bg-red-700 w-7 h-7 rounded-full m-2'></div>
          <div>
          <h1 className='font-extrabold text-white p-1'>HIGH RISK AREAS <br /> 12</h1>
          </div>
      </div>
      <div className='w-70 h-20 bg-blue-950 rounded m-5 flex items-center pl-5'>
          <div className='bg-orange-500 w-7 h-7 rounded-full m-2'></div>
          <div>
          <h1 className='font-extrabold text-white p-1'>Recent Landslides <br /> 4</h1>
          </div>
      </div>
      <div className='w-70 h-20 bg-blue-950 rounded m-5 flex items-center pl-5'>
          <div className='bg-green-600 w-7 h-7 rounded-full m-2 items-center flex'> <CloudRain></CloudRain></div>
          <div>
          <h1 className='font-extrabold text-white p-1'>Rainfall  <br /> 12</h1>
          </div>
      </div>
      <div className='w-70 h-20 bg-blue-950 rounded m-5 flex items-center pl-5'>
          <div className='bg-[#b59824] w-7 h-7 rounded-full m-2'></div>
          <div>
          <h1 className='font-extrabold text-white p-1'>Soil Moisture Level <br /> 52%</h1>
          </div>
      </div>
     
    </div>
    <div className="relative z-0 w-full h-[500px] mt-4">
     <MapComponent></MapComponent>
     </div>
    </div>
    </>
  )
}

export default Landslide