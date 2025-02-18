use axum::{routing::post, Router, Json, extract::State};
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::{Device, Tensor};  // ✅ Correct Import
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tokio::net::TcpListener;
use tracing::{info, error};
use tracing_subscriber;

#[derive(Debug, Deserialize)]
struct InputData {
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct OutputData {
    result: Vec<f32>,
}

struct ModelState {
    device: Device,
}

async fn predict(
    State(state): State<Arc<Mutex<ModelState>>>,
    Json(input): Json<InputData>,
) -> Json<OutputData> {
    let model = state.lock().await;
    match Tensor::from_vec(input.values.clone(), (input.values.len(),), &model.device) {
        Ok(tensor) => match tensor.to_vec1() {
            Ok(output) => {
                info!("Prediction successful");
                Json(OutputData { result: output })
            },
            Err(e) => {
                error!("Failed to process tensor: {:?}", e);
                Json(OutputData { result: vec![] })
            }
        },
        Err(e) => {
            error!("Tensor creation failed: {:?}", e);
            Json(OutputData { result: vec![] })
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = Arc::new(Mutex::new(ModelState {
        device: Device::Cpu,
    }));

    let app = Router::new()
        .route("/predict", post(predict))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = TcpListener::bind("0.0.0.0:8000").await.unwrap();
    info!("Running AI API on http://localhost:8000");
    axum::serve(listener, app).await.unwrap();
}
