## Aim:

To develop an AI-based classroom attendance system using face recognition that automatically detects and recognizes students from a classroom image or camera feed and marks their attendance as Present or Absent.

## System Requirements:
Programming Language: Python
Libraries: OpenCV, Face Recognition, NumPy
Input: Student reference images and classroom/group photo or live camera feed
Output: Recognized faces with attendance status
Storage: CSV file for attendance records
Platform: Google Colab / Python environment

## Features:
Detects multiple faces in a classroom image.
Recognizes registered students using face recognition.
Marks recognized students as Present.
Marks registered students who are not detected as Absent.
Identifies unregistered faces as Unknown.
Displays bounding boxes around detected faces.
Generates an attendance summary.
Saves attendance records in a CSV file.
Provides attendance statistics such as total students, present, absent, and attendance percentage.

## Program 

```
# ============================================================
# INSTALL REQUIRED PACKAGE
# ============================================================

!pip install -q dlib-bin
!pip install -q face-recognition


# ============================================================
# FACE RECOGNITION ATTENDANCE SYSTEM
# ============================================================

import cv2
import face_recognition
import os
import csv
import shutil
from datetime import datetime
from google.colab.patches import cv2_imshow


# ============================================================
# SETTINGS
# ============================================================

STUDENTS_DIR = "/content/students"
CLASSROOM_IMAGE = "/content/Students.jpeg"
ATTENDANCE_FILE = "/content/attendance.csv"

TOLERANCE = 0.50


# ============================================================
# 1. SETUP
# ============================================================

print("==========================================")
print("CHECKING FILES AND SETTING UP DIRECTORIES")
print("==========================================")

os.makedirs(STUDENTS_DIR, exist_ok=True)


# Copy reference images into students folder
student_images = [
    "Spoorthi.jpeg",
    "Susmitha.jpeg"
]


for filename in student_images:

    source = f"/content/{filename}"
    destination = f"{STUDENTS_DIR}/{filename}"

    if os.path.exists(source):

        shutil.copy(source, destination)

        print(f"[INFO] Added {filename} to students/")

    else:

        print(f"[WARNING] {filename} not found!")


# Show files
print("\nFiles in students folder:")

student_files = os.listdir(STUDENTS_DIR)

for file in student_files:
    print(" -", file)


# Check classroom image
if not os.path.exists(CLASSROOM_IMAGE):

    raise FileNotFoundError(
        f"Classroom image '{CLASSROOM_IMAGE}' not found!"
    )

print(
    f"\n[SUCCESS] Classroom image "
    f"'{CLASSROOM_IMAGE}' found."
)


# ============================================================
# 2. LOAD STUDENT FACES
# ============================================================

print("\n=========================================")
print("LOADING STUDENT FACES")
print("=========================================")

known_faces = []
known_names = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png"
)


for filename in student_files:

    if not filename.lower().endswith(valid_extensions):
        continue

    path = os.path.join(
        STUDENTS_DIR,
        filename
    )

    print(f"\n[INFO] Processing: {filename}")

    try:

        image = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(
            image
        )

        if len(encodings) == 0:

            print(
                f"[WARNING] No face detected in {filename}"
            )

            continue

        known_faces.append(encodings[0])

        name = os.path.splitext(filename)[0]

        known_names.append(name)

        print(
            f"[SUCCESS] Registered: {name}"
        )

    except Exception as e:

        print(
            f"[ERROR] Could not process {filename}"
        )

        print(e)


# ============================================================
# 3. REGISTERED STUDENTS
# ============================================================

print("\n=========================================")
print("REGISTERED STUDENTS")
print("=========================================")

print(
    f"Total students loaded: {len(known_names)}"
)

for name in known_names:
    print(" -", name)


if len(known_faces) == 0:

    raise ValueError(
        "No student faces were loaded. "
        "Make sure Spoorthi.jpeg and Susmitha.jpeg "
        "contain clear faces."
    )


# ============================================================
# 4. LOAD CLASSROOM IMAGE
# ============================================================

print("\n=========================================")
print("LOADING CLASSROOM IMAGE")
print("=========================================")

classroom = cv2.imread(
    CLASSROOM_IMAGE
)

if classroom is None:

    raise ValueError(
        "Unable to read Students.jpeg"
    )

print(
    "[SUCCESS] Classroom image loaded."
)


# ============================================================
# 5. CONVERT IMAGE
# ============================================================

rgb_classroom = cv2.cvtColor(
    classroom,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 6. DETECT FACES
# ============================================================

print("\n=========================================")
print("DETECTING FACES")
print("=========================================")

face_locations = face_recognition.face_locations(
    rgb_classroom
)

face_encodings = face_recognition.face_encodings(
    rgb_classroom,
    face_locations
)

print(
    f"[INFO] Detected {len(face_encodings)} face(s)."
)


# ============================================================
# 7. INITIALIZE ATTENDANCE
# ============================================================

present_students = []

absent_students = known_names.copy()


# ============================================================
# 8. RECOGNIZE STUDENTS
# ============================================================

print("\n=========================================")
print("RECOGNIZING STUDENTS")
print("=========================================")


for face_encoding, face_location in zip(
    face_encodings,
    face_locations
):

    face_distances = face_recognition.face_distance(
        known_faces,
        face_encoding
    )

    best_match_index = face_distances.argmin()

    name = "Unknown"

    if face_distances[best_match_index] <= TOLERANCE:

        name = known_names[best_match_index]

        distance = face_distances[
            best_match_index
        ]

        print(
            f"[MATCH] {name} "
            f"(distance: {distance:.3f})"
        )

        if name not in present_students:

            present_students.append(name)

        if name in absent_students:

            absent_students.remove(name)

    else:

        print(
            "[UNKNOWN] Unknown face detected."
        )


    # ========================================================
    # DRAW FACE BOX
    # ========================================================

    top, right, bottom, left = face_location

    if name == "Unknown":

        color = (0, 0, 255)

    else:

        color = (0, 255, 0)


    cv2.rectangle(
        classroom,
        (left, top),
        (right, bottom),
        color,
        2
    )


    # ========================================================
    # DISPLAY NAME
    # ========================================================

    cv2.putText(
        classroom,
        name,
        (left, max(top - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


# ============================================================
# 9. RECOGNITION RESULT
# ============================================================

print("\n=========================================")
print("RECOGNITION RESULT")
print("=========================================")

cv2_imshow(classroom)


# ============================================================
# 10. SAVE ATTENDANCE
# ============================================================

print("\n=========================================")
print("SAVING ATTENDANCE")
print("=========================================")

timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

with open(
    ATTENDANCE_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Student Name",
        "Status",
        "Timestamp"
    ])

    for name in known_names:

        if name in present_students:

            status = "Present"

        else:

            status = "Absent"

        writer.writerow([
            name,
            status,
            timestamp
        ])


print(
    "[SUCCESS] Attendance saved to attendance.csv"
)


# ============================================================
# 11. ATTENDANCE SUMMARY
# ============================================================

print("\n=========================================")
print("ATTENDANCE SUMMARY")
print("=========================================")

for name in known_names:

    if name in present_students:

        print(
            f"✅ {name}: PRESENT"
        )

    else:

        print(
            f"❌ {name}: ABSENT"
        )


# ============================================================
# 12. ATTENDANCE STATISTICS
# ============================================================

total_students = len(known_names)

total_present = len(present_students)

total_absent = (
    total_students -
    total_present
)


print("\n=========================================")
print("ATTENDANCE STATISTICS")
print("=========================================")

print(
    "Total Students :",
    total_students
)

print(
    "Present        :",
    total_present
)

print(
    "Absent         :",
    total_absent
)


if total_students > 0:

    percentage = (
        total_present /
        total_students
    ) * 100

    print(
        f"Attendance     : "
        f"{percentage:.2f}%"
    )


print(
    "\n[DONE] Attendance system completed!"
)
```

## Output

<img width="713" height="627" alt="image" src="https://github.com/user-attachments/assets/e8f2dff7-6544-4908-8118-e75b1def0515" />

<img width="453" height="573" alt="image" src="https://github.com/user-attachments/assets/ba0f2fad-22df-4502-8dee-3dd1430ca350" />

<img width="286" height="238" alt="image" src="https://github.com/user-attachments/assets/def8296c-b2c5-4f5d-8396-dcf9af95ba42" />


## Result 

The AI-based Classroom Attendance System was successfully implemented using face recognition. The system detects faces from the classroom image, recognizes registered students, automatically marks them as Present or Absent, identifies unknown faces, and stores the attendance details in a CSV file.
