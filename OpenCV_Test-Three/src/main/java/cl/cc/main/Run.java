package cl.cc.main;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_core.cvRound;
import static org.bytedeco.javacpp.opencv_core.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import org.bytedeco.javacpp.opencv_objdetect;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_ROUGH_SEARCH;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_FIND_BIGGEST_OBJECT;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_SCALE_IMAGE;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FrameGrabber;

/**
 *
 * @author CyberCastle
 * 
 */
public class Run {

    public static void main(String[] args) throws Exception {

        String faceClassifierPath = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
        String leftEyeClassifierPath = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_lefteye.xml";
        String rightEyeClassifierPath = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_righteye.xml";
        String noseEyeClassifierPath = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml";
        String mouthEyeClassifierPath = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml";

        // Preload the opencv_objdetect module to work around a known bug.
        Loader.load(opencv_objdetect.class);

        CascadeClassifier faceClassifier = new CascadeClassifier(faceClassifierPath);

        // Establecemos el origen del stream a procesar
        FrameGrabber grabber = FrameGrabber.createDefault(0);

        grabber.setImageHeight(720);
        grabber.setImageWidth(1280);
        grabber.start();

        // Captura de la imagen de la cámara
        IplImage grabbedImage = grabber.grab();

        // Creamos la ventana de visualización
        CanvasFrame frame = new CanvasFrame("Sonria!!!!!! :-)", CanvasFrame.getDefaultGamma() / grabber.getGamma());

        while (frame.isVisible() && (grabbedImage = grabber.grab()) != null) {

            Mat img = new Mat(grabbedImage, true);
            Mat gray = new Mat();
            Mat smallImg = new Mat((img.rows() / 2), cvRound(img.cols() / 2), CV_8UC1);
            cvtColor(img, gray, CV_BGR2GRAY);
            resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
            equalizeHist( smallImg, smallImg );
            Rect faces = new opencv_core.Rect();
            faceClassifier.detectMultiScale(smallImg, faces, 1.1, 1, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_SCALE_IMAGE, new Size(0, 0), new Size(250, 250));

            Point center = new Point();
            center.x(cvRound(faces.x() + faces.width() * 0.5));
            center.y(cvRound(faces.y() + faces.height() * 0.5));
            int radius = cvRound((faces.width() + faces.height()) * 0.35);

            // Recortando la cara, despreciando el resto de la imagen.
            Point x1 = new Point((center.x() - radius + 20), (center.y() - radius));
            Point x2 = new Point((center.x() + radius - 20), (center.y() + radius));
            Rect roi = new Rect(x1.x(), x1.y(), (x2.x() - x1.x()), (x2.y() - x1.y()));

            //smallImg.apply(myROI);
            //circle(smallImg, center, radius, new opencv_core.Scalar(0, 255, 0, 255), 2, 8, 0);
            rectangle(smallImg, x1, x2, new opencv_core.Scalar(255, 255, 0, 255), 2, 8, 0);

            if ((0 <= roi.x() && 0 <= roi.width() && roi.x() + roi.width() <= 0 && 0 <= roi.y() && 0 <= roi.height() && roi.y() + roi.height() <= 0)) {
                smallImg = smallImg.apply(roi);
            }
            frame.showImage(smallImg);
        }
        
        frame.dispose();
        grabber.stop();
    }

}
