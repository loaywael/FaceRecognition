# FaceRecognition
Face Detection and Recognition  | under development
</br></br></br>

- # Architecture
<img src="assets/pipeline/architecture.jpg">
</br></br></br>

- # Pipeline
    - Resizeing raw sample images  to 96 x 96
    <table>
        <tr>
            <td><img align="center" src="assets/pipeline/src_224x265.png"></td>
            <td><img src="assets/pipeline/src2_204x206.png"></td>
        </tr>
        <tr>
            <td><img align="center" src="assets/pipeline/src_96x96.png"></td>
            <td><img src="assets/pipeline/src2_96x96.png"></td>
        </tr>
    </table>
    </br></br>
        
    <h2>Resized batch of 128x96x96x1<h2>
    <td><img align="center" src="assets/pipeline/raw_face_batch.png"></td></br>
    <h2>Resized with landmarks overlayed<h2>
    <td><img src="assets/pipeline/face_keypoints_batch.png"></td></br>
    <h2>Resized batch of 64x48x48x1 of expressions dataset<h2>
    <td><img src="assets/pipeline/face_expressions_batch.png" style="width: 100%"></td></br>
