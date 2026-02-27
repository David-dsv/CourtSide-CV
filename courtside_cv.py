#!/usr/bin/env python
"""
CourtSide-CV - Suivi fluide et continu de la balle
Combine ByteTrack + Interpolation + Smoothing
"""

import os
import cv2
import argparse
import numpy as np
import subprocess
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from collections import deque, defaultdict
import logging
from scipy import interpolate
from trackers import ByteTrackTracker
import supervision as sv
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BallTracker:
    """Tracker de balle avec interpolation et lissage"""

    def __init__(self):
        # Modèle YOLO26 fine-tuné pour tennis ball
        yolo26_path = Path(__file__).parent / "training" / "yolo26_ball_best.pt"
        if yolo26_path.exists():
            logger.info(f"Chargement du modèle YOLO26 : {yolo26_path}")
            self.ball_model = YOLO(str(yolo26_path))
            self.is_yolo26 = True  # classe 0 = tennis_ball
        else:
            # Fallback: modèle HuggingFace
            logger.info("YOLO26 non trouvé, fallback sur HuggingFace")
            weights_path = hf_hub_download(
                repo_id="Davidsv/CourtSide-Computer-Vision-v1",
                filename="model.pt"
            )
            self.ball_model = YOLO(weights_path)
            self.is_yolo26 = False  # classe 1 = tennis_ball

        # ByteTrack tracker (roboflow/trackers 2.2.0)
        self.byte_tracker = ByteTrackTracker(
            lost_track_buffer=30,
            track_activation_threshold=0.05,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=0.1,
            frame_rate=30
        )

        # Tracking
        self.tracks = {}
        self.frame_idx = 0
        self.all_positions = []  # [(frame, x, y, conf)]

        # Paramètres optimisés
        self.conf_thresh = 0.05
        self.smooth_window = 5
        self.max_interpolate_gap = 30  # Max 1 sec à 30fps

    def process_batch(self, frames):
        """Process un batch de frames pour le tracking"""
        positions = []

        for i, frame in enumerate(frames):
            self.frame_idx = i

            # Détection seule (sans tracker intégré ultralytics)
            # YOLO26 fine-tuné : classe 0 = tennis_ball
            # Modèle HuggingFace : classe 1 = tennis_ball
            results = self.ball_model(
                source=frame,
                conf=self.conf_thresh,
                classes=[0] if self.is_yolo26 else [1],
                imgsz=640,
                iou=0.5,
                verbose=False
            )

            # Extraire position avec ByteTrack (trackers 2.2.0)
            ball_pos = None
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=confs,
                    class_id=cls_ids
                )
                tracked = self.byte_tracker.update(detections)

                if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
                    best_idx = tracked.confidence.argmax()
                    x1, y1, x2, y2 = tracked.xyxy[best_idx]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    conf = float(tracked.confidence[best_idx])
                    ball_pos = (cx, cy, conf)

            positions.append((i, ball_pos))

            if i % 100 == 0:
                detected = sum(1 for _, p in positions if p is not None)
                pct = (detected / len(positions)) * 100 if positions else 0
                logger.info(f"Batch progress: {i}/{len(frames)} | Detection rate: {pct:.1f}%")

        return positions

    def interpolate_missing(self, positions):
        """Interpoler les positions manquantes"""
        # Séparer les positions détectées
        detected_frames = []
        detected_x = []
        detected_y = []

        for frame_idx, pos in positions:
            if pos is not None:
                detected_frames.append(frame_idx)
                detected_x.append(pos[0])
                detected_y.append(pos[1])

        if len(detected_frames) < 2:
            logger.warning("Pas assez de détections pour interpoler")
            return positions

        # Créer les fonctions d'interpolation
        fx = interpolate.interp1d(detected_frames, detected_x,
                                  kind='linear', fill_value='extrapolate')
        fy = interpolate.interp1d(detected_frames, detected_y,
                                  kind='linear', fill_value='extrapolate')

        # Interpoler les positions manquantes
        interpolated = []
        for frame_idx, pos in positions:
            if pos is None:
                # Vérifier si on est dans une zone interpolable
                prev_detected = max([f for f in detected_frames if f < frame_idx], default=-999)
                next_detected = min([f for f in detected_frames if f > frame_idx], default=999)

                if (frame_idx - prev_detected <= self.max_interpolate_gap and
                    next_detected - frame_idx <= self.max_interpolate_gap):
                    # Interpoler
                    ix = float(fx(frame_idx))
                    iy = float(fy(frame_idx))
                    interpolated.append((frame_idx, (ix, iy, 0.0)))  # conf=0 pour interpolé
                else:
                    interpolated.append((frame_idx, None))
            else:
                interpolated.append((frame_idx, pos))

        return interpolated

    def smooth_trajectory(self, positions):
        """Lisser la trajectoire avec filtre médian"""
        smoothed = []

        for i, (frame_idx, pos) in enumerate(positions):
            if pos is None:
                smoothed.append((frame_idx, None))
                continue

            # Fenêtre pour le lissage
            window_start = max(0, i - self.smooth_window // 2)
            window_end = min(len(positions), i + self.smooth_window // 2 + 1)

            # Collecter les positions dans la fenêtre
            window_x = []
            window_y = []
            for j in range(window_start, window_end):
                if positions[j][1] is not None:
                    window_x.append(positions[j][1][0])
                    window_y.append(positions[j][1][1])

            if window_x:
                # Médiane pour lisser
                smooth_x = np.median(window_x)
                smooth_y = np.median(window_y)
                conf = pos[2] if len(pos) > 2 else 0.0
                smoothed.append((frame_idx, (smooth_x, smooth_y, conf)))
            else:
                smoothed.append((frame_idx, pos))

        return smoothed


class VideoProcessor:
    """Processeur vidéo avec tracking, pose et annotation"""

    def __init__(self):
        self.tracker = BallTracker()
        self.person_model = YOLO('yolov8m.pt')
        self.pose_model = YOLO('yolov8m-pose.pt')  # Modèle de pose pour squelette

        # Connexions du squelette COCO (17 keypoints)
        self.skeleton_connections = [
            (5, 6),   # Épaules
            (5, 7),   # Épaule gauche -> coude gauche
            (7, 9),   # Coude gauche -> poignet gauche
            (6, 8),   # Épaule droite -> coude droit
            (8, 10),  # Coude droit -> poignet droit
            (5, 11),  # Épaule gauche -> hanche gauche
            (6, 12),  # Épaule droite -> hanche droite
            (11, 12), # Hanches
            (11, 13), # Hanche gauche -> genou gauche
            (13, 15), # Genou gauche -> cheville gauche
            (12, 14), # Hanche droite -> genou droit
            (14, 16), # Genou droit -> cheville droite
            (0, 1),   # Nez -> œil gauche
            (0, 2),   # Nez -> œil droit
            (1, 3),   # Œil gauche -> oreille gauche
            (2, 4),   # Œil droit -> oreille droite
        ]

    def draw_skeleton(self, frame, keypoints, conf_threshold=0.5):
        """Dessine le squelette sur la frame"""
        # Couleurs pour le squelette
        joint_color = (0, 255, 0)  # Vert pour les joints
        bone_color = (0, 255, 255)  # Cyan pour les os

        # Dessiner les connexions (os)
        for connection in self.skeleton_connections:
            kp1_idx, kp2_idx = connection

            if kp1_idx < len(keypoints) and kp2_idx < len(keypoints):
                kp1 = keypoints[kp1_idx]
                kp2 = keypoints[kp2_idx]

                # Vérifier la confiance
                if len(kp1) > 2 and len(kp2) > 2:
                    if kp1[2] > conf_threshold and kp2[2] > conf_threshold:
                        pt1 = (int(kp1[0]), int(kp1[1]))
                        pt2 = (int(kp2[0]), int(kp2[1]))

                        # Ligne avec effet glow
                        cv2.line(frame, pt1, pt2, bone_color, 3, cv2.LINE_AA)
                        cv2.line(frame, pt1, pt2, (128, 255, 255), 2, cv2.LINE_AA)

        # Dessiner les points (articulations)
        for i, kp in enumerate(keypoints):
            if len(kp) > 2 and kp[2] > conf_threshold:
                x, y = int(kp[0]), int(kp[1])

                # Point avec effet glow
                cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 4, joint_color, -1)

                # Points spéciaux pour les mains (indices 9 et 10)
                if i in [9, 10]:  # Poignets
                    cv2.circle(frame, (x, y), 8, (255, 0, 255), 2)  # Magenta pour les mains

    def process_video(self, video_path, start_time, end_time, output_dir='output',
                       player1_name='PLAYER 1', player2_name='PLAYER 2'):
        os.makedirs(output_dir, exist_ok=True)

        # Parse times
        start_sec = sum(x * int(t) for x, t in zip([60, 1], start_time.split(':')))
        end_sec = sum(x * int(t) for x, t in zip([60, 1], end_time.split(':')))
        duration = end_sec - start_sec

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Aller au début
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_sec * fps)

        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING VIDEO")
        logger.info(f"Segment: {start_time} - {end_time} ({duration}s)")
        logger.info(f"FPS: {fps}, Resolution: {width}x{height}")
        logger.info(f"{'='*60}\n")

        # Phase 1: Charger toutes les frames
        logger.info("Phase 1: Loading frames...")
        frames = []
        max_frames = duration * fps

        for i in range(int(max_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

            if i % 100 == 0:
                logger.info(f"Loaded {i}/{max_frames} frames")

        logger.info(f"Loaded {len(frames)} frames total")

        # Phase 2: Tracking
        logger.info("\nPhase 2: Ball tracking...")
        positions = self.tracker.process_batch(frames)

        # Stats avant interpolation
        detected = sum(1 for _, p in positions if p is not None)
        logger.info(f"Initial detection rate: {detected}/{len(positions)} ({detected*100/len(positions):.1f}%)")

        # Phase 3: Interpolation
        logger.info("\nPhase 3: Interpolating missing positions...")
        positions = self.tracker.interpolate_missing(positions)

        # Stats après interpolation
        filled = sum(1 for _, p in positions if p is not None)
        logger.info(f"After interpolation: {filled}/{len(positions)} ({filled*100/len(positions):.1f}%)")

        # Phase 4: Smoothing
        logger.info("\nPhase 4: Smoothing trajectory...")
        positions = self.tracker.smooth_trajectory(positions)

        # Phase 5: Rendering
        logger.info("\nPhase 5: Rendering output video...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_video = f"{output_dir}/temp_{timestamp}.mp4"
        final_output = f"{output_dir}/courtside_{timestamp}.mp4"

        # mp4v fonctionne mieux avec OpenCV, ffmpeg convertira en H.264
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        # Trail pour effet visuel
        trail = deque(maxlen=30)

        # Compteurs de coups
        stats_coups = {
            'player1': {'name': player1_name, 'fh': 0, 'bh': 0},
            'player2': {'name': player2_name, 'fh': 0, 'bh': 0}
        }

        # Instructions
        logger.info("\n" + "="*50)
        logger.info("CONTRÔLES MANUELS:")
        logger.info(f"  {player1_name}: A = Coup Droit | Q = Revers")
        logger.info(f"  {player2_name}: P = Coup Droit | L = Revers")
        logger.info("  ESPACE = Pause/Play | ESC = Quitter")
        logger.info("="*50 + "\n")

        cv2.namedWindow('Tennis Analysis', cv2.WINDOW_NORMAL)
        paused = False

        for frame_idx, (frame, (_, ball_pos)) in enumerate(zip(frames, positions)):
            annotated = frame.copy()

            # Balle
            if ball_pos is not None:
                cx, cy, conf = int(ball_pos[0]), int(ball_pos[1]), ball_pos[2]

                # Ajouter au trail
                trail.append((cx, cy))

                # Couleur selon confiance
                if conf > 0.1:
                    ball_color = (0, 255, 255)  # Cyan - détecté
                    glow_color = (128, 255, 255)
                else:
                    ball_color = (255, 200, 0)  # Orange - interpolé
                    glow_color = (255, 220, 128)

                # Effet glow
                for r in [20, 15, 10]:
                    alpha = 0.3 * (20 - r) / 20
                    overlay = annotated.copy()
                    cv2.circle(overlay, (cx, cy), r, glow_color, -1)
                    annotated = cv2.addWeighted(annotated, 1-alpha, overlay, alpha, 0)

                # Balle principale
                cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), -1)
                cv2.circle(annotated, (cx, cy), 6, ball_color, -1)

                # Trail
                for i in range(1, len(trail)):
                    thickness = max(1, (i * 3) // len(trail))
                    alpha = i / len(trail)
                    color = tuple(int(c * alpha) for c in (255, 255, 0))
                    cv2.line(annotated, trail[i-1], trail[i], color, thickness, cv2.LINE_AA)

            # Détection de pose pour squelette
            pose_results = self.pose_model(frame, conf=0.25, imgsz=640, verbose=False)

            if pose_results[0].keypoints is not None:
                # Obtenir les keypoints (max 2 personnes pour les 2 joueurs)
                all_keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                all_confidences = pose_results[0].keypoints.conf.cpu().numpy() if pose_results[0].keypoints.conf is not None else None

                # Limiter à 2 personnes
                for person_idx in range(min(2, len(all_keypoints))):
                    keypoints = all_keypoints[person_idx]

                    # Ajouter les confidences si disponibles
                    if all_confidences is not None:
                        keypoints_with_conf = []
                        for kp_idx in range(len(keypoints)):
                            x, y = keypoints[kp_idx]
                            conf = all_confidences[person_idx][kp_idx] if person_idx < len(all_confidences) else 0.5
                            keypoints_with_conf.append([x, y, conf])
                    else:
                        keypoints_with_conf = [[kp[0], kp[1], 1.0] for kp in keypoints]

                    # Dessiner le squelette
                    self.draw_skeleton(annotated, keypoints_with_conf, conf_threshold=0.3)

            # Boîtes des joueurs (optionnel, plus discret avec le squelette)
            person_results = self.person_model(frame, conf=0.5, classes=[0], imgsz=640, verbose=False)
            if person_results[0].boxes is not None:
                boxes = person_results[0].boxes.xyxy[:2]  # Max 2 joueurs
                for box in boxes:
                    x1, y1, x2, y2 = box.tolist()
                    # Rectangle très discret
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                                (0, 150, 0), 1, cv2.LINE_AA)

            # Barre de status en bas - CourtSide-CV
            cv2.rectangle(annotated, (0, height-45), (width, height), (0, 0, 0), -1)

            # Logo CourtSide-CV à gauche
            cv2.putText(annotated, "CourtSide-CV", (15, height-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Status de la balle au centre
            if ball_pos is not None:
                status = "TRACKING" if conf > 0.1 else "PREDICTED"
                color_status = (0, 255, 255) if conf > 0.1 else (255, 200, 0)
                cv2.putText(annotated, f"Ball: {status}", (width//2 - 60, height-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2, cv2.LINE_AA)

            # Compteur de coups (en haut à gauche et à droite)
            # Joueur 1 (gauche)
            cv2.rectangle(annotated, (10, 50), (180, 110), (0, 0, 0), -1)
            cv2.rectangle(annotated, (10, 50), (180, 110), (0, 255, 255), 2)
            cv2.putText(annotated, stats_coups['player1']['name'], (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"FH: {stats_coups['player1']['fh']}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"BH: {stats_coups['player1']['bh']}", (100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1, cv2.LINE_AA)

            # Joueur 2 (droite)
            cv2.rectangle(annotated, (width-180, 50), (width-10, 110), (0, 0, 0), -1)
            cv2.rectangle(annotated, (width-180, 50), (width-10, 110), (0, 255, 255), 2)
            cv2.putText(annotated, stats_coups['player2']['name'], (width-170, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"FH: {stats_coups['player2']['fh']}", (width-170, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"BH: {stats_coups['player2']['bh']}", (width-90, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1, cv2.LINE_AA)

            # Afficher pour contrôle en temps réel
            cv2.imshow('Tennis Analysis', annotated)

            # Gestion des touches
            wait_time = 1 if not paused else 0
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('a'):  # Joueur 1 FH
                stats_coups['player1']['fh'] += 1
                logger.info(f"{player1_name} FH: {stats_coups['player1']['fh']}")
            elif key == ord('q'):  # Joueur 1 BH
                stats_coups['player1']['bh'] += 1
                logger.info(f"{player1_name} BH: {stats_coups['player1']['bh']}")
            elif key == ord('p'):  # Joueur 2 FH
                stats_coups['player2']['fh'] += 1
                logger.info(f"{player2_name} FH: {stats_coups['player2']['fh']}")
            elif key == ord('l'):  # Joueur 2 BH
                stats_coups['player2']['bh'] += 1
                logger.info(f"{player2_name} BH: {stats_coups['player2']['bh']}")
            elif key == ord(' '):  # Pause/Play
                paused = not paused
                logger.info("PAUSED" if paused else "PLAYING")
            elif key == 27:  # ESC
                logger.info("Stopped by user")
                break

            # Si en pause, attendre sans avancer
            while paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(' '):
                    paused = False
                    logger.info("PLAYING")
                    break
                elif key == 27:
                    paused = False
                    break

            out.write(annotated)

            if frame_idx % 100 == 0:
                logger.info(f"Rendered {frame_idx}/{len(frames)} frames")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Afficher stats finales des coups
        logger.info(f"\n{'='*50}")
        logger.info("STATS FINALES:")
        p1 = stats_coups['player1']
        p2 = stats_coups['player2']
        logger.info(f"  {p1['name']}: FH={p1['fh']}, BH={p1['bh']}")
        logger.info(f"  {p2['name']}: FH={p2['fh']}, BH={p2['bh']}")
        logger.info(f"{'='*50}\n")

        # Phase 6: Ajouter le son et reencoder en H.264
        logger.info("\nPhase 6: Adding audio and ensuring compatibility...")

        # Extraire l'audio du segment original
        audio_temp = f"{output_dir}/audio_temp_{timestamp}.aac"
        cmd_audio = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_sec), '-t', str(duration),
            '-vn', '-acodec', 'aac', '-b:a', '192k',
            audio_temp, '-y', '-loglevel', 'error'
        ]
        subprocess.run(cmd_audio, check=True)

        # Combiner vidéo + audio et convertir en H.264
        cmd = [
            'ffmpeg',
            '-i', temp_video,
            '-i', audio_temp,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '22',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'copy',
            '-shortest',
            '-movflags', '+faststart',
            final_output, '-y', '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)

        # Nettoyer les fichiers temporaires
        os.remove(temp_video)
        os.remove(audio_temp)

        # Stats finales
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ VIDEO READY!")
        logger.info(f"Output: {final_output}")
        logger.info(f"Duration: {duration}s")
        logger.info(f"Coverage: {filled*100/len(positions):.1f}% frames with ball")
        logger.info(f"{'='*60}\n")

        return final_output


def main():
    parser = argparse.ArgumentParser(description='CourtSide-CV Tennis Tracking')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--start', default='0:00', help='Start time (mm:ss)')
    parser.add_argument('--end', default='0:30', help='End time (mm:ss)')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--player1', default='PLAYER 1', help='Name of player 1 (left)')
    parser.add_argument('--player2', default='PLAYER 2', help='Name of player 2 (right)')

    args = parser.parse_args()

    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║              CourtSide-CV  Tennis Tracking                    ║
    ╚════════════════════════════════════════════════════════════════╝

    Features:
    - ByteTrack object tracking for ball
    - Pose estimation with skeleton visualization
    - Intelligent interpolation for missing frames
    - Trajectory smoothing
    - Visual effects (glow, trail, skeleton)

    Processing...
    """)

    processor = VideoProcessor()
    output = processor.process_video(args.video, args.start, args.end, args.output,
                                      player1_name=args.player1, player2_name=args.player2)

    print(f"""
    Output: {output}
    """)


if __name__ == '__main__':
    main()