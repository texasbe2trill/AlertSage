# MITRE ATT&CK® Attribution

This project makes limited, research-focused use of the **MITRE ATT&CK®** knowledge base.

Specifically, the synthetic incident generator and modeling notebooks:

- Reference **ATT&CK technique IDs** (for example, `T1078`, `T1190`, `T1486`, `T1566`),
- Include **lightly paraphrased language** inspired by public ATT&CK technique descriptions,
- Use these techniques to make simulated adversary behavior more realistic in narrative text fields.

No proprietary or internal threat intel is used; all references are drawn from publicly available ATT&CK content and are intended for educational and portfolio purposes only.

## Trademarks and license

MITRE ATT&CK® and ATT&CK® are registered trademarks of **The MITRE Corporation**.

ATT&CK data is provided by The MITRE Corporation and is licensed under the
**Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)**.

For full details, see the official MITRE ATT&CK site and licensing information.

## How this project uses ATT&CK

Within this repository, ATT&CK references primarily appear in:

- The synthetic data generator (`generator/generate_cyber_incidents.py`), where certain
  narratives add clauses such as "This behavior aligns with MITRE ATT&CK technique T1486"
  to emulate analyst-style writeups.
- Modeling and evaluation notebooks, which occasionally refer to relevant tactics or
  techniques when discussing example incidents or patterns (for example, ransomware
  encryption mapped to `T1486`, phishing mapped to `T1566`, or web exploitation mapped
  to `T1190`).

These references are **contextual aids** to make the simulated incidents feel more
SOC-realistic and to help readers connect incident categories to canonical ATT&CK
techniques. They are **not** intended to be authoritative mappings and should not be
used as the sole basis for production detection logic.

## Scope and limitations

- The dataset is **synthetic** and only loosely aligned to ATTATT&CK many edge cases and
  real-world nuances are intentionally simplified.
- Technique IDs included in narratives are examples rather than exhaustive coverage.
- This repository does **not** redistribute the full ATT&CK corpus; it only uses short
  paraphrased clauses and technique IDs.

If you use this project in your own work, please ensure that any further use of
ATT&CK content continues to respect MITRE's licensing and trademark guidance.

## MITRE ATT&CK Integration in AlertSage

AlertSage maps MITRE ATT&CK techniques at the **incident classification level**. Once an incident description is analyzed and classified (for example, phishing or malware), a predefined set of relevant ATT&CK technique IDs is associated with that incident type to provide standardized adversary context.
This mapping is implemented in the AlertSage UI layer (`ui_premium.py`) using a static lookup table. The approach is intentionally deterministic and explainable, ensuring that SOC analysts see consistent MITRE context for a given incident type without relying on opaque or probabilistic technique inference.

### Example Incident: Phishing

**Incident Description:**  
A user reports receiving an email impersonating internal IT support that requests password reset confirmation through a malicious link.

**Predicted Incident Type:** Phishing

**Mapped MITRE ATT&CK Techniques:**
- `T1566` – Phishing
- `T1598` – Phishing for Information

**SOC Analyst Context:**  
These techniques indicate an initial access attempt focused on credential harvesting. Analysts can prioritize email gateway logs, user click activity, and authentication events during triage.

### Example Incident: Malware

**Incident Description:**  
An endpoint begins executing an unknown binary downloaded from an external source, followed by unauthorized encryption of local files.

**Predicted Incident Type:** Malware

**Mapped MITRE ATT&CK Techniques:**
- `T1059` – Command and Scripting Interpreter
- `T1105` – Ingress Tool Transfer
- `T1486` – Data Encrypted for Impact

**SOC Analyst Context:**  
This pattern suggests malware execution with potential ransomware behavior. Analysts should review process execution logs, file system changes, and outbound network connections.

### Example Incident: Web Attack

**Incident Description:**  
Multiple suspicious requests are observed against a public-facing web application, exploiting an unpatched vulnerability to execute unauthorized server-side commands.

**Predicted Incident Type:** Web Attack

**Mapped MITRE ATT&CK Techniques:**
- `T1190` – Exploit Public-Facing Application
- `T1059.007` – JavaScript

**SOC Analyst Context:**  
This activity indicates an initial access attempt via application-layer exploitation. Analysts should investigate web server logs, error traces, and recent application changes.

## Incident Type to MITRE ATT&CK Reference

| Incident Type | Technique ID | Technique Name | ATT&CK Tactic |
|--------------|-------------|----------------|---------------|
| Phishing | T1566 | Phishing | Initial Access |
| Phishing | T1598 | Phishing for Information | Reconnaissance |
| Malware | T1059 | Command and Scripting Interpreter | Execution |
| Malware | T1105 | Ingress Tool Transfer | Command and Control |
| Malware | T1486 | Data Encrypted for Impact | Impact |
| Web Attack | T1190 | Exploit Public-Facing Application | Initial Access |
| Web Attack | T1059.007 | JavaScript | Execution |

## Example MITRE-Enriched JSON Output
```
{
  "incident_type": "phishing",
  "confidence": "High",
  "mitre_techniques": [
    {
      "technique_id": "T1566",
      "name": "Phishing",
      "tactic": "Initial Access"
    },
    {
      "technique_id": "T1598",
      "name": "Phishing for Information",
      "tactic": "Reconnaissance"
    }
  ]
}

 