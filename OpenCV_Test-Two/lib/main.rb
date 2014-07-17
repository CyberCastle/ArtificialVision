require 'opencv'
include OpenCV

capture = CvCapture.open(0)

data = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
detector = CvHaarClassifierCascade::load(data)
window = GUI::Window.new('Face detection')

while capture.grab do
  img = capture.retrieve
  detector.detect_objects(img).each do |region|
    color = CvColor::Blue
    img.rectangle! region.top_left, region.bottom_right, :color => color
  end
  window.show(img)
  GUI::wait_key
end