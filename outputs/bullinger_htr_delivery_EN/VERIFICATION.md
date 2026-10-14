# Delivery verification

The packaged official-only command completed successfully offline on 14 October 2026 using the existing local environment and cached model. All five metric summaries exactly matched the saved results across 357 paragraphs and 327 official place annotations. This verifies reproduction; it is not a new independent evaluation.

Python sources and notebook code cells passed syntax checks. Package file hashes were checked. No HTR training or GPU runtime was started.

GitHub preparation changes only the model-cache lookup to the identical pinned snapshot path. Original full evaluation was completed before this packaging adjustment. No fresh full evaluation was run during publication.
