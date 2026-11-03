package com.edgefloraxai.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class RemedyResult(
    val matched: Boolean,
    val matchType: String,
    val immediateActions: List<String>,
    val culturalControl: List<String>,
    val biologicalControl: List<String>,
    val chemicalControl: List<String>,
    val prevention: List<String>,
    val safetyNotes: List<String>,
    val sourceTitle: String,
    val sourceOrganization: String
)

class RemedyEngine(
    private val context: Context
) {

    private val remedies: JSONArray

    init {

        val jsonText =
            context.assets
                .open("remedy_knowledge_base.json")
                .bufferedReader()
                .use {
                    it.readText()
                }

        val root =
            JSONObject(jsonText)

        remedies =
            root.getJSONArray(
                "recommendations"
            )
    }

    fun getRemedy(
        disease: String,
        flower: String
    ): RemedyResult {

        // =========================
        // HEALTHY
        // =========================

        if (disease == "Healthy") {

            return RemedyResult(
                matched = true,
                matchType = "healthy",

                immediateActions =
                    listOf(
                        "No disease treatment required."
                    ),

                culturalControl =
                    emptyList(),

                biologicalControl =
                    emptyList(),

                chemicalControl =
                    emptyList(),

                prevention =
                    listOf(
                        "Continue regular plant monitoring."
                    ),

                safetyNotes =
                    emptyList(),

                sourceTitle =
                    "",

                sourceOrganization =
                    ""
            )
        }

        // =========================
        // FIND GENERAL MATCH
        // =========================

        var generalIndex =
            -1

        // =========================
        // SEARCH REMEDIES
        // =========================

        for (
        i in 0 until remedies.length()
        ) {

            val item =
                remedies.getJSONObject(i)

            val itemDisease =
                item.optString(
                    "disease"
                )

            val itemFlower =
                item.optString(
                    "flower"
                )

            // =========================
            // EXACT FLOWER MATCH
            // =========================

            if (
                itemDisease == disease &&
                itemFlower == flower
            ) {

                return convertResult(
                    item,
                    "flower_specific"
                )
            }

            // =========================
            // GENERAL MATCH
            // =========================

            if (
                generalIndex == -1 &&
                itemDisease == disease &&
                itemFlower ==
                "General flowering plants"
            ) {

                generalIndex = i
            }
        }

        // =========================
        // USE GENERAL MATCH
        // =========================

        if (generalIndex != -1) {

            return convertResult(
                remedies.getJSONObject(
                    generalIndex
                ),
                "general"
            )
        }

        // =========================
        // NO VERIFIED RECOMMENDATION
        // =========================

        return RemedyResult(
            matched = false,

            matchType =
                "no_verified_recommendation",

            immediateActions =
                emptyList(),

            culturalControl =
                emptyList(),

            biologicalControl =
                emptyList(),

            chemicalControl =
                emptyList(),

            prevention =
                emptyList(),

            safetyNotes =
                emptyList(),

            sourceTitle =
                "",

            sourceOrganization =
                ""
        )
    }

    // =========================
    // CONVERT JSON ITEM
    // =========================

    private fun convertResult(
        item: JSONObject,
        matchType: String
    ): RemedyResult {

        return RemedyResult(

            matched = true,

            matchType =
                matchType,

            immediateActions =
                getStringList(
                    item,
                    "immediate_actions"
                ),

            culturalControl =
                getStringList(
                    item,
                    "cultural_control"
                ),

            biologicalControl =
                getStringList(
                    item,
                    "biological_control"
                ),

            chemicalControl =
                getStringList(
                    item,
                    "chemical_control"
                ),

            prevention =
                getStringList(
                    item,
                    "prevention"
                ),

            safetyNotes =
                getStringList(
                    item,
                    "safety_notes"
                ),

            sourceTitle =
                item.optString(
                    "source_title"
                ),

            sourceOrganization =
                item.optString(
                    "source_organization"
                )
        )
    }

    // =========================
    // READ JSON STRING ARRAY
    // =========================

    private fun getStringList(
        item: JSONObject,
        key: String
    ): List<String> {

        val array =
            item.optJSONArray(key)
                ?: return emptyList()

        val result =
            mutableListOf<String>()

        for (
        i in 0 until array.length()
        ) {

            result.add(
                array.optString(i)
            )
        }

        return result
    }

    // =========================
    // CLOSE
    // =========================

    fun close() {
        // No resources need to be closed.
    }
}