Frozen regression inputs from the qualified K32 compiled application:
- source-request.json and selected-plan.json from application/
- precision-policy.json from compile-inputs/
- region_002 through region_004 C/header bytes from application/c-package/regions/

These are copied unchanged, except that source-request.json, region_004 and the
ReLU plan row come from the ordered-select K32 application, since the sign-clamp
route no longer exists; the plan is the adapter's plan for these frozen inputs. The source contains code/exponent input signatures,
not weight payloads. This fixture checks unchanged selection and host emission;
it does not establish equality of freshly emitted device programs.
