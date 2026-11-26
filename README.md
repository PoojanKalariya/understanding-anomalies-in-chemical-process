# Chemical Process Anomaly Explorer

An interactive Streamlit dashboard for exploring how sensor measurements respond to documented disturbances in a batch distillation process.

## Motivation

Understanding sensor behavior during abnormal chemical-process conditions helps researchers inspect documented disturbances, compare affected measurements, and form evidence-based hypotheses before building detection or diagnosis models.

## Dataset

This deployment package contains a focused subset of the Batch Distillation Dataset:

- six operating points: OP10, OP17, OP18, OP22, OP29, and OP31;
- 13 recordings: seven normal and six anomalous runs;
- 11 selected sensors sampled approximately once per second; and
- YAML annotations describing anomaly timing, category, disturbance, and documented local effects.

The recordings come from a laboratory batch distillation process involving butan-1-ol, propan-2-ol, and water. Each anomalous CSV is paired with its YAML annotation by operating point and filename stem.

## Features

- Interactive nested treemap grouping anomalies by operating point, with equal weight per recording-scoped anomaly
- Eleven-sensor response overview with valid YAML intervals highlighted
- Independent event-context and Full Run controls for the sensor overview and detailed raw-measurement inspection
- Full-recording visualization for recurrent anomalies
- Optional full-recording overlay with a normal run from the same operating point, aligned by elapsed time

## Technologies

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- PyYAML

The nested treemap uses a local Streamlit custom component with a bundled Plotly.js asset; no external CDN is required.

## Installation and local execution

Requires Python 3.11 or later.

```bash
git clone <your-repository-url>
cd chemical-process-anomaly-explorer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

With `uv`:

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
streamlit run app.py
```

## Live demo

Deployment URL: **TBD — add the Streamlit Community Cloud URL after deployment.**

## Methodology and limitations

- YAML annotations guide anomaly investigation; they do not prove sensor-level causation.
- Normal and anomalous runs are independent experiments and are not treated as physically synchronized.
- Unresolved annotation timestamps are retained and reported, never silently corrected or shaded as valid intervals.
- The multi-sensor overview applies documented per-sensor robust baseline-relative scaling (median/MAD) for visual comparison. When no pre-event sample exists, it explicitly uses a same-operating-point normal recording as a scaling reference; this does not align experiments in time.
- The detailed chart retains original sensor values.

## Dataset attribution and license

The included data subset is redistributed under the source record's **CC BY 4.0** license. Please retain attribution when reusing it.

Arweiler, J., Jungjohann, I., Muraleedharan, A., Leitte, H., Burger, J., Münnemann, K., Jirasek, F., & Hasse, H. (2025). *Batch Distillation Data for Developing Machine Learning Anomaly Detection Methods*. Zenodo. [https://doi.org/10.5281/zenodo.17395544](https://doi.org/10.5281/zenodo.17395544)

## Project status

This release is the exploratory sensor-visualization stage of the project. Model comparison, automated anomaly detection, and explainability are planned future work; they are not included here.
