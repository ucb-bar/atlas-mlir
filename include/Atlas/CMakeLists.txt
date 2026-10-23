add_mlir_dialect(AtlasOps atlas)

set(ATLAS_MODEL_VOCABULARY ${PROJECT_BINARY_DIR}/include/Atlas/AtlasModelVocabulary.h)
add_custom_command(
  OUTPUT ${ATLAS_MODEL_VOCABULARY}
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/atlas_model_schema.py
    --vocabulary ${PROJECT_SOURCE_DIR}/tools/model-timing-vocabulary.json
    --output ${ATLAS_MODEL_VOCABULARY}
  DEPENDS ${PROJECT_SOURCE_DIR}/tools/atlas_model_schema.py
          ${PROJECT_SOURCE_DIR}/tools/model-timing-vocabulary.json
          ${PROJECT_SOURCE_DIR}/tools/model-timing.lock.json
  VERBATIM)
add_custom_target(AtlasModelVocabularyGen DEPENDS ${ATLAS_MODEL_VOCABULARY})
