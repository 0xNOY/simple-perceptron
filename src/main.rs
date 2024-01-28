use std::path::PathBuf;

use plotters::prelude::*;
use serde::{Deserialize, Serialize};

type Float = f32;

fn step(x: Float) -> Float {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[derive(Clone, Debug)]
struct SimplePerceptron<const N: usize> {
    weights: [Float; N], // シナプス結合荷重
    threshold: Float,    // 閾値
}

impl<const N: usize> SimplePerceptron<N> {
    fn predict(&self, inputs: [Float; N]) -> Float {
        // 単純パーセプトロンの出力を計算する
        let mut sum = -self.threshold;
        for i in 0..N {
            sum += self.weights[i] * inputs[i];
        }
        step(sum)
    }

    fn fit(&mut self, data: &TeacherData<N>, learning_rate: Float, fix_threshold: bool) {
        // 誤り訂正学習法に基づいてシナプス結合荷重を修正する
        let prediction = self.predict(data.inputs);
        let error = data.target - prediction;
        if !fix_threshold {
            // 閾値を固定しない場合は閾値も修正する
            self.threshold -= learning_rate * error;
        }
        for i in 0..N {
            self.weights[i] += learning_rate * error * data.inputs[i];
        }
    }
}

#[derive(Serialize, Deserialize)]
struct TeacherData<const N: usize> {
    #[serde(with = "serde_arrays")]
    inputs: [Float; N],
    target: Float,
}

fn load_dataset<const N: usize>(path: &PathBuf) -> Vec<TeacherData<N>> {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    serde_json::from_reader(reader).unwrap()
}

fn train_perceptron<const N: usize>(
    perceptron: &mut SimplePerceptron<N>,
    teacher_dataset: &[TeacherData<N>],
    epoch: isize, // 0未満であれば収束するまで学習を続ける
    learning_rate: Float,
    fix_threshold: bool,
) -> Vec<SimplePerceptron<N>> {
    let mut perceptron_history = vec![perceptron.clone()];

    for i in 0.. {
        if i == epoch {
            return perceptron_history;
        }

        for data in teacher_dataset {
            // 単純パーセプトロンの学習
            perceptron.fit(data, learning_rate, fix_threshold);

            // パーセプトロンの状態を記録
            perceptron_history.push(perceptron.clone());

            // 学習が完了したかどうかを判定
            if !teacher_dataset
                .iter()
                .any(|data| perceptron.predict(data.inputs) != data.target)
            {
                return perceptron_history;
            }
        }
    }

    unreachable!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const MY_CLASS_NUM: usize = 2;
    const MY_STUDENT_NUM: usize = 28;

    let result_dir = PathBuf::from("results");
    let figures_dir = result_dir.join("figures");
    let teacher_dataset_path = PathBuf::from("training_data.json");

    // ハイパーパラメータの設定
    const INPUT_NUM: usize = 7;
    let fix_bias = true;
    let learning_rates = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
    let init_bias = 0.5;
    let init_weights = [(MY_STUDENT_NUM * (2 * MY_CLASS_NUM - 3)) as Float / 25.0; INPUT_NUM];

    // データセットの読み込み
    let teacher_dataset = load_dataset(&teacher_dataset_path);

    let mut weight_histories = vec![Vec::new(); learning_rates.len()];

    for (i, rl) in learning_rates.iter().enumerate() {
        // 単純パーセプトロンの初期化
        let mut perceptron = SimplePerceptron {
            weights: init_weights,
            threshold: init_bias,
        };

        // 単純パーセプトロンの学習
        let perceptron_history =
            train_perceptron(&mut perceptron, &teacher_dataset, -1, *rl, fix_bias);

        weight_histories[i] = perceptron_history.iter().map(|p| p.weights).collect();
    }

    // 決定された重みをcsvとして出力する
    std::fs::create_dir_all(&result_dir)?;
    let mut writer = csv::Writer::from_path(result_dir.join("weights.csv"))?;
    for (rl, history) in learning_rates.iter().zip(weight_histories.iter()) {
        let w = history
            .last()
            .unwrap()
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>();
        writer.write_record(std::iter::once(&rl.to_string()).chain(w.iter()))?;
    }

    // 学習回数をcsvとして出力する
    let mut writer = csv::Writer::from_path(result_dir.join("iteration.csv"))?;
    for (rl, h) in learning_rates.iter().zip(weight_histories.iter()) {
        writer.write_record([rl.to_string(), h.len().to_string()])?;
    }

    // 学習の履歴をプロットする
    std::fs::create_dir_all(&figures_dir)?;
    let min_weight = weight_histories
        .iter()
        .map(|h| {
            h.iter()
                .map(|w| w.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
        })
        .flatten()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_weight = weight_histories
        .iter()
        .map(|h| {
            h.iter()
                .map(|w| w.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
        })
        .flatten()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    for (rl, history) in learning_rates.iter().zip(weight_histories.iter()) {
        let fig_path = figures_dir.join(format!("weight_history_rl{}.png", rl));
        let root = BitMapBackend::new(&fig_path, (1600, 1200)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .x_label_area_size(95)
            .y_label_area_size(115)
            .build_cartesian_2d(0..history.len(), *min_weight..*max_weight)?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Iteration")
            .y_desc("Weight")
            .label_style(("sans-serif", 54))
            .draw()?;

        for i in 0..INPUT_NUM {
            let color = Palette99::pick(i)
                .to_rgba()
                .mix(0.8)
                .stroke_width(1)
                .filled();
            chart
                .draw_series(
                    LineSeries::new(history.iter().enumerate().map(|(x, w)| (x, w[i])), color)
                        .point_size(3),
                )?
                .label(format!("w_{}", i + 1))
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 48, y)], color.stroke_width(8))
                });
        }

        chart
            .configure_series_labels()
            .label_font(("sans-serif", 48))
            .position(SeriesLabelPosition::UpperLeft)
            .legend_area_size(60)
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(())
}
