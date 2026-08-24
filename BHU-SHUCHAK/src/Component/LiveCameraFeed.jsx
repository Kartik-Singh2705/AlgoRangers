import React from 'react'

const LiveCameraFeed = () => {

  const images=[
   " https://static01.nyt.com/images/2012/09/24/world/asia/24-Sikkim-landslide-IndiaInk/24-Sikkim-landslide-IndiaInk-blog480.jpg",
   "https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iVpGC8lhb49g/v0/-1x-1.webp",
   "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjt9feQp5HMJn9nlAvQEp9gWRO6baJXx_lldpgILCh5gdZ3g4Ne9HC7PA&s=10",
  ]

  return (
    <>
    <h1 className='ml-10 font-extrabold text-2xl'>Live Camera Feeds</h1>
    <div className='flex gap-6 m-5 flex-wrap'>
      {images.map((e)=>(
          <img className='rounded-4xl w-90' src={e} alt="" />
      ))}     
    </div>
    </>
  )
}

export default LiveCameraFeed