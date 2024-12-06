import '../../Styles/main.css'
import '../../Styles/feature-showcase.css'
import '../../Styles/features-overview.css'
import '../../Styles/call-to-action.css'
import '../../Styles/footer.css'
import { FeatureShowcase } from "../FeatureShowcase"
import { CallToAction } from "../CallToAction"
import Button from '../../Style/Button'
import { Footer } from '../../Layout/Footer/Footer'
import Header from '../../Layout/Header/Header'
import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="min-h-screen bg-[rgb(16,17,36)] text-white">
      
      <Header/>

      <main className="px-8">
        <section className="py-24 max-w-6xl mx-auto">
          <div className="space-y-6 mb-12">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <div className="w-2 h-2 rounded-full bg-[rgb(82,97,255)]" />
              ADVANCED GESTURE RECOGNITION FOR DEVELOPERS
            </div>
            <h1 className="text-5xl font-medium leading-tight">
              Revolutionize user interaction<br />
              with AI-powered gesture detection
            </h1>
          </div>

          <div className="grid gap-4 mb-12">
            
              <span className="flex items-center text-xl gap-2 font-medium">
                
                <span className="w-1.5 h-1.5 rounded-full bg-[rgb(82,97,255)]" />                  
                Facial Expression Analysis
                 
              </span>

              <span className="flex items-center text-xl gap-2 font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-[rgb(82,97,255)]" />
                Body Pose Estimation
                 
              </span>

              <span className="flex items-center text-xl gap-2 font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-[rgb(82,97,255)]" />
                Real-time Motion Tracking
                 
              </span>
          </div>

          <Link to="/videoupload">
          <Button className="bg-[rgb(82,97,255)] hover:bg-[rgb(82,97,255)]/90 text-white rounded-full px-8 py-6">
              GET STARTED
            </Button>
          </Link>

          <div className="mt-32 space-y-8">
            <h2 className="text-4xl font-medium text-center">See GestureAI in action</h2>
            <div className="relative w-full aspect-video bg-black/20 rounded-lg overflow-hidden">
              <div className="absolute inset-0 flex items-center justify-center">
                <GestureAnimation />
              </div>
            </div>
          </div>

          <div className="mt-16 grid grid-cols-3 gap-4">
            <GestureBubble
              gesture="Swipe right"
              avatar="/placeholder.svg?height=40&width=40"
            />
            <GestureBubble
              gesture="Pinch to zoom"
              avatar="/placeholder.svg?height=40&width=40"
            />
            <GestureBubble
              gesture="Two-finger rotate"
              avatar="/placeholder.svg?height=40&width=40"
            />
          </div>
        </section>

        <section className="py-24 max-w-6xl mx-auto">
          <div className="text-center space-y-6 mb-16">
            <div className="text-sm text-gray-400">HOW WE CAN HELP</div>
            <h2 className="text-5xl font-medium">Struggling with complex gesture integration?</h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Discover use cases where GestureAI can enhance your applications<br />
              and improve user experience without complex coding
            </p>
          </div>

          <FeatureShowcase />
        </section>

        <CallToAction />
        <Footer />
      </main>
    </div>
  )
}

function GestureAnimation() {
  return (
    <div className="flex items-center justify-center w-full h-full">
      <div className="relative w-32 h-32">
        <div className="absolute inset-0 border-4 border-[rgb(82,97,255)] rounded-full animate-ping" />
        <div className="absolute inset-4 border-4 border-[rgb(82,97,255)] rounded-full animate-ping" style={{ animationDelay: '0.5s' }} />
        <div className="absolute inset-8 border-4 border-[rgb(82,97,255)] rounded-full animate-ping" style={{ animationDelay: '1s' }} />
      </div>
    </div>
  )
}

function GestureBubble({ gesture, avatar }: { gesture: string; avatar: string }) {
  return (
    <div className="flex items-center gap-4 p-4 rounded-lg bg-white/5">
      <img src={avatar} alt="Avatar" className="w-10 h-10 rounded-full" />
      <p className="text-sm text-gray-300">{gesture}</p>
    </div>
  )
}

