# Audit package completeness

Both `oasis180_protocol_summary.md` and `tensor_build_trace.md` are **non-empty on this machine** (3013 and 5011 bytes respectively in the original `oasis_protocol_provenance_audit_20260713/` directory). The zero-byte files received downstream were a transfer/export artifact, not a local data-loss event.

| path                               | exists   |   size_bytes |   line_count | sha256                                                           | status   | repair_action                                 |
|:-----------------------------------|:---------|-------------:|-------------:|:-----------------------------------------------------------------|:---------|:----------------------------------------------|
| oasis180_scan_crosswalk.csv        | True     |       286959 |          181 | 67061881cdef12be5043ed361bf2fcd4340abc125c6e350a76418f0d8f5bfa82 | INTACT   | NO_ACTION_REQUIRED_ORIGINAL_INTACT            |
| oasis180_protocol_distribution.csv | True     |          603 |            6 | bb2d647adc567c970de6739ffbf88fea5ff6dc0f58e0ee03c7888094f28b7a2b | INTACT   | NO_ACTION_REQUIRED_ORIGINAL_INTACT            |
| oasis180_protocol_summary.md       | True     |         3013 |           62 | 50326e3985152284bcd975630cb57e25253d3c0dff1ba85b328f8ffd3c208c03 | INTACT   | VERIFIED_COPY_ORIGINAL_INTACT                 |
| tensor_build_trace.md              | True     |         5011 |           48 | 3fc9920aaee5bfe76f67b49e27717153019c061a395e9df70647db405ad75168 | INTACT   | VERIFIED_COPY_PLUS_ENRICHMENT_ORIGINAL_INTACT |
| published_claim_reconciliation.md  | True     |         2607 |           25 | 51f45d004eff104069be8466d0d035bd17f0becd7032010f78a48af35af329b4 | INTACT   | NO_ACTION_REQUIRED_ORIGINAL_INTACT            |
| manuscript_safe_wording.md         | True     |         1405 |           13 | 9620ab7244077135cbb9435b89a81e9af1eba7ebd9d989841f6785c5f99c0bd8 | INTACT   | NO_ACTION_REQUIRED_ORIGINAL_INTACT            |
| unresolved_observations.csv        | True     |           76 |            1 | dfc8358a9d73307a9888772af73496295c27d3d1a83b332ae0d85d54701dec0e | INTACT   | NO_ACTION_REQUIRED_ORIGINAL_INTACT            |
| verification_gates.json            | True     |         1946 |           38 | fa1446eaa904de2f2fa44f6b3af70654ffb16e1471f14fa5c913f6b9cbb3ec42 | INTACT   | NO_ACTION_REQUIRED_ORIGINAL_INTACT            |
| command_log.json                   | True     |        12334 |          248 | 5b4a68fe3fc20de4ec4cb97612b68b5ba4cc75c87da9ae4d96e36d3b273bd248 | INTACT   | NO_ACTION_REQUIRED_ORIGINAL_INTACT            |
