package com.edgefloraxai.app

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {

            var selectedImageUri by remember {
                mutableStateOf<Uri?>(null)
            }

            var statusText by remember {
                mutableStateOf("Select a plant image")
            }

            val imagePicker =
                rememberLauncherForActivityResult(
                    ActivityResultContracts.GetContent()
                ) { uri ->

                    if (uri != null) {
                        selectedImageUri = uri

                        statusText =
                            "Image selected.\nTap Analyze Plant."
                    }
                }

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),

                horizontalAlignment =
                    Alignment.CenterHorizontally,

                verticalArrangement =
                    Arrangement.Center
            ) {

                Text(
                    text = "EdgeFloraXAI",
                    style =
                        MaterialTheme
                            .typography
                            .headlineMedium
                )

                Spacer(
                    modifier =
                        Modifier.height(30.dp)
                )

                // =========================
                // SELECT IMAGE BUTTON
                // =========================

                Button(
                    onClick = {
                        imagePicker.launch("image/*")
                    }
                ) {
                    Text(
                        text = "Select Plant Image"
                    )
                }

                Spacer(
                    modifier =
                        Modifier.height(20.dp)
                )

                // =========================
                // ANALYZE BUTTON
                // =========================

                Button(
                    onClick = {

                        val uri =
                            selectedImageUri

                        if (uri == null) {

                            statusText =
                                "Please select an image first."

                            return@Button
                        }

                        try {

                            statusText =
                                "Analyzing..."

                            // =========================
                            // READ IMAGE
                            // =========================

                            val stream =
                                contentResolver
                                    .openInputStream(uri)

                            val bitmap =
                                BitmapFactory
                                    .decodeStream(stream)

                            stream?.close()

                            if (bitmap == null) {

                                statusText =
                                    "Could not read image."

                                return@Button
                            }

                            // =========================
                            // YOLO
                            // =========================

                            statusText =
                                "Detecting plant..."

                            val yolo =
                                YoloDetector(
                                    this@MainActivity
                                )

                            val plantResults =
                                yolo.detect(bitmap)

                            yolo.close()

                            if (
                                plantResults.isEmpty()
                            ) {

                                bitmap.recycle()

                                statusText =
                                    "Image not confidently recognized.\n\n" +
                                            "Please capture a clear photo of the plant or leaf."

                                return@Button
                            }

                            val plant =
                                plantResults[0]

                            val plantConfidence =
                                plant.confidence * 100f

                            // =========================
                            // YOLO CONFIDENCE SAFEGUARD
                            // =========================

                            if (
                                plantConfidence < 20.0f
                            ) {

                                bitmap.recycle()

                                statusText =
                                    "Image not confidently recognized.\n\n" +
                                            "Plant detection confidence: " +
                                            "%.2f%%".format(
                                                plantConfidence
                                            ) +
                                            "\n\n" +
                                            "Please capture a clear photo of the plant or leaf."

                                return@Button
                            }

                            // =========================
                            // MOBILEVIT
                            // =========================

                            statusText =
                                "Classifying disease..."

                            val classifier =
                                MobileVitClassifier(
                                    this@MainActivity
                                )

                            val disease =
                                classifier.classify(
                                    bitmap
                                )

                            classifier.close()

                            val diseaseConfidence =
                                disease.confidence * 100f

                            // =========================
                            // DISEASE CONFIDENCE SAFEGUARD
                            // =========================

                            if (
                                diseaseConfidence < 60.0f
                            ) {

                                bitmap.recycle()

                                statusText =
                                    "Image not confidently recognized.\n\n" +
                                            "Plant: ${plant.className}\n" +
                                            "Plant confidence: %.2f%%\n\n".format(
                                                plantConfidence
                                            ) +
                                            "Disease confidence: %.2f%%\n\n".format(
                                                diseaseConfidence
                                            ) +
                                            "Please capture a clear photo of the diseased leaf or plant."

                                return@Button
                            }

                            // =========================
                            // U-NET SEGMENTATION
                            // =========================

                            statusText =
                                "Analyzing diseased region..."

                            val segmenter =
                                UnetSegmenter(
                                    this@MainActivity
                                )

                            val segmentation =
                                segmenter.segment(
                                    bitmap,
                                    diseaseName =
                                        disease.className
                                )

                            segmenter.close()

                            // =========================
                            // REMEDY ENGINE
                            // =========================

                            statusText =
                                "Preparing recommendation..."

                            val remedyEngine =
                                RemedyEngine(
                                    this@MainActivity
                                )

                            val remedy =
                                remedyEngine.getRemedy(
                                    disease.className,
                                    plant.className
                                )

                            remedyEngine.close()

                            // =========================
                            // FINAL RESULT
                            // =========================

                            bitmap.recycle()

                            statusText =
                                buildResultText(
                                    plantName =
                                        plant.className,
                                    plantConfidence =
                                        plantConfidence,
                                    diseaseName =
                                        disease.className,
                                    diseaseConfidence =
                                        diseaseConfidence,
                                    segmentation =
                                        segmentation,
                                    remedy =
                                        remedy
                                )

                        } catch (e: Exception) {

                            statusText =
                                """
Error:
${e.javaClass.simpleName}

${e.message}
                                """.trimIndent()
                        }
                    }
                ) {
                    Text(
                        text = "Analyze Plant"
                    )
                }

                Spacer(
                    modifier =
                        Modifier.height(30.dp)
                )

                Text(
                    text = statusText
                )
            }
        }
    }

    // =========================
    // BUILD RESULT TEXT
    // =========================

    private fun buildResultText(
        plantName: String,
        plantConfidence: Float,
        diseaseName: String,
        diseaseConfidence: Float,
        segmentation: SegmentationResult,
        remedy: RemedyResult
    ): String {

        val builder =
            StringBuilder()

        // =========================
        // PLANT
        // =========================

        builder.append(
            "Plant: $plantName\n\n"
        )

        builder.append(
            "Plant confidence: " +
                    "%.2f".format(
                        plantConfidence
                    ) +
                    "%\n\n"
        )

        // =========================
        // DISEASE
        // =========================

        builder.append(
            "Disease: $diseaseName\n\n"
        )

        builder.append(
            "Disease confidence: " +
                    "%.2f".format(
                        diseaseConfidence
                    ) +
                    "%\n\n"
        )

        // =========================
        // DISEASE AREA
        // =========================

        builder.append(
            "Disease area: " +
                    "%.2f".format(
                        segmentation.diseaseAreaPercent
                    ) +
                    "%\n\n"
        )

        // =========================
        // SEVERITY
        // =========================

        builder.append(
            "Severity: " +
                    "%.2f".format(
                        segmentation.severityPercent
                    ) +
                    "%\n\n"
        )

        builder.append(
            "Severity band:\n" +
                    segmentation.severityBand +
                    "\n\n"
        )

        // =========================
        // RECOMMENDATIONS
        // =========================

        builder.append(
            "RECOMMENDATIONS\n\n"
        )

        // =========================
        // NO VERIFIED REMEDY
        // =========================

        if (!remedy.matched) {

            builder.append(
                "No verified recommendation " +
                        "is available for this " +
                        "disease and plant combination.\n"
            )

            return builder.toString()
        }

        // =========================
        // HEALTHY
        // =========================

        if (
            remedy.matchType == "healthy"
        ) {

            builder.append(
                "No disease treatment required.\n\n"
            )

            if (
                remedy.prevention.isNotEmpty()
            ) {

                builder.append(
                    "Prevention:\n"
                )

                for (
                item in remedy.prevention
                ) {

                    builder.append(
                        "• $item\n"
                    )
                }
            }

            return builder.toString()
        }

        // =========================
        // IMMEDIATE ACTIONS
        // =========================

        if (
            remedy.immediateActions.isNotEmpty()
        ) {

            builder.append(
                "Immediate actions:\n"
            )

            for (
            item in remedy.immediateActions
            ) {

                builder.append(
                    "• $item\n"
                )
            }

            builder.append("\n")
        }

        // =========================
        // CULTURAL CONTROL
        // =========================

        if (
            remedy.culturalControl.isNotEmpty()
        ) {

            builder.append(
                "Cultural control:\n"
            )

            for (
            item in remedy.culturalControl
            ) {

                builder.append(
                    "• $item\n"
                )
            }

            builder.append("\n")
        }

        // =========================
        // BIOLOGICAL CONTROL
        // =========================

        if (
            remedy.biologicalControl.isNotEmpty()
        ) {

            builder.append(
                "Biological control:\n"
            )

            for (
            item in remedy.biologicalControl
            ) {

                builder.append(
                    "• $item\n"
                )
            }

            builder.append("\n")
        }

        // =========================
        // CHEMICAL CONTROL
        // =========================

        if (
            remedy.chemicalControl.isNotEmpty()
        ) {

            builder.append(
                "Chemical control:\n"
            )

            for (
            item in remedy.chemicalControl
            ) {

                builder.append(
                    "• $item\n"
                )
            }

            builder.append("\n")
        }

        // =========================
        // PREVENTION
        // =========================

        if (
            remedy.prevention.isNotEmpty()
        ) {

            builder.append(
                "Prevention:\n"
            )

            for (
            item in remedy.prevention
            ) {

                builder.append(
                    "• $item\n"
                )
            }

            builder.append("\n")
        }

        // =========================
        // SAFETY
        // =========================

        if (
            remedy.safetyNotes.isNotEmpty()
        ) {

            builder.append(
                "Safety:\n"
            )

            for (
            item in remedy.safetyNotes
            ) {

                builder.append(
                    "• $item\n"
                )
            }

            builder.append("\n")
        }

        // =========================
        // SOURCE
        // =========================

        if (
            remedy.sourceTitle.isNotEmpty()
        ) {

            builder.append(
                "Source:\n"
            )

            builder.append(
                remedy.sourceTitle
            )

            builder.append("\n")

            builder.append(
                remedy.sourceOrganization
            )
        }

        return builder.toString()
    }
}