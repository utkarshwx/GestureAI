'use client'

import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'
import Home from './components/Home/Home'
import VideoInteration from './components/VideoInteraction/VideoInteration'

export default function App() {


  return (
    <>
      <div className="App">
        <Router>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/videoupload" element={<VideoInteration />} />
          </Routes>
        </Router>
      </div>

    </>
  )


}
