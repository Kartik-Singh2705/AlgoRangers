import React, { useContext, useState } from 'react'
import { Search } from "lucide-react";
import { User } from 'lucide-react';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react'
import { ChevronDownIcon } from '@heroicons/react/20/solid'
import { MapPin } from 'lucide-react';
import { AppContext } from '../Component/AppContext';

const NavBar = () => {

  const{location,setLocation,setCoordinates}=useContext(AppContext);

  

  const handleClick=(value)=>{
    setLocation(value.v);
    setCoordinates(value.c)
  }
  return (
    
    <div className='shadow-xl  w-screen h-20 border-r-2 flex items-center justify-between'>
      <h1 className="font-extrabold text-transparent bg-clip-text bg-linear-to-r from-[#498b13] via-[#a09c2d] to-[#D2691E] m-4 text-2xl">
          BHU-SUCHACK
      </h1>

    <div className='flex'>
       <Menu as="div" className="relative inline-block mt-8">
      <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-xs inset-ring-1 inset-ring-gray-300 hover:bg-gray-50">
       {location}
        <ChevronDownIcon aria-hidden="true" className="-mr-1 size-5 text-gray-400" />
      </MenuButton>

      <MenuItems
        transition
        className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg outline-1 outline-black/5 transition data-closed:scale-95 data-closed:transform data-closed:opacity-0 data-enter:duration-100 data-enter:ease-out data-leave:duration-75 data-leave:ease-in"
      >
        <div className="py-1">
          <MenuItem className='flex gap-2'>
            <a
              href="#"
              className="block px-4 py-2 text-sm text-gray-700 data-focus:bg-gray-100 data-focus:text-gray-900 data-focus:outline-hidden"
              onClick={(e)=>handleClick({ v: "Current Location", c: [28.6139, 77.2090] })}
            >
             <MapPin></MapPin> Current Location
            </a>
          </MenuItem>
          <MenuItem>
            <a
              href="#"
              className="block px-4 py-2 text-sm text-gray-700 data-focus:bg-gray-100 data-focus:text-gray-900 data-focus:outline-hidden"
              onClick={(e)=>handleClick({v:"Gangtok",c:[27.3314, 88.6130]})}
            >
             Gangtok
            </a>
          </MenuItem>
          <MenuItem>
            <a
              href="#"
              className="block px-4 py-2 text-sm text-gray-700 data-focus:bg-gray-100 data-focus:text-gray-900 data-focus:outline-hidden"
              onClick={()=>{handleClick({v:"Kalimpong",c:[27.0346, 88.6308]})}}
            >
             Kalimpong
            </a>
          </MenuItem>
          <form action="#" method="POST">
            <MenuItem>
              <a
               
                className="block w-full px-4 py-2 text-left text-sm text-gray-700 data-focus:bg-gray-100 data-focus:text-gray-900 data-focus:outline-hidden"
                onClick={(e)=>{handleClick({v:"Itanagar",c:[27.0, 94.0]})}}
              >
                Itanagar
              </a>
            </MenuItem>
          </form>
        </div>
      </MenuItems>
    </Menu>
    <a className='m-10' href=""><Search></Search>
    </a>
    <a href="" className='mt-10 mr-5'> <User /></a>
    </div>
      
    </div>
  )
}

export default NavBar