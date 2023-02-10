use std::io::stdout;
use std::time::Instant;

use dynamic_weighted_index::DynamicWeightedIndex;
use pcg_rand::Pcg64;
use rand::distributions::Distribution;
use rand::Rng;
use rand::SeedableRng;

fn benchmark_random_increase(n: usize, g: usize, samples: usize) {
    let mut rng = Pcg64::from_entropy();

    let mut weights = Vec::new();

    let mut dyn_index: DynamicWeightedIndex<f64> = DynamicWeightedIndex::new(n);
    for i in 0..n {
        weights.push(1.0);
        dyn_index.set_weight(i, 1.0);
    }

    let steps = 100 * n;
    let substeps = steps / g;
    let mut t = 0;
    while (t < steps) {
        let start = Instant::now();
        for i in 0..samples {
            let s = dyn_index.sample(&mut rng).unwrap();
        }
        let runtime = start.elapsed();
        println!("WeightedIndex RandomIncrease [n: {}] Time elapsed: {}ms", t, runtime.as_millis());

        for i in 0..substeps {
            let u = rng.gen_range(0..n);
            weights[u] += 1.0;
            dyn_index.set_weight(u, weights[u]);
        }
        t += substeps;
    }
}

fn benchmark_polya_urn(n: usize, g: usize, samples: usize) {
    let mut rng = Pcg64::from_entropy();

    let mut weights = Vec::new();

    let mut dyn_index: DynamicWeightedIndex<f64> = DynamicWeightedIndex::new(n);
    for i in 0..n {
        weights.push(1.0);
        dyn_index.set_weight(i, 1.0);
    }

    let steps = 100 * n;
    let substeps = steps / g;
    let mut t = 0;
    while (t < steps) {
        let start = Instant::now();
        for i in 0..samples {
            let s = dyn_index.sample(&mut rng).unwrap();
        }
        let runtime = start.elapsed();
        println!("WeightedIndex PolyaUrn [n: {}] Time elapsed: {}ms", t, runtime.as_millis());

        for i in 0..substeps {
            let u = dyn_index.sample(&mut rng).unwrap();
            weights[u] += 1.0;
            dyn_index.set_weight(u, weights[u]);
        }
        t += substeps;
    }
}

fn benchmark_single_increase(n: usize, g: usize, samples: usize) {
    let mut rng = Pcg64::from_entropy();

    let mut weights = Vec::new();

    let mut dyn_index: DynamicWeightedIndex<f64> = DynamicWeightedIndex::new(n);
    for i in 0..n {
        weights.push(1.0);
        dyn_index.set_weight(i, 1.0);
    }

    let steps = 100 * n;
    let substeps = steps / g;
    let mut t = 0;
    while (t < steps) {
        let start = Instant::now();
        for i in 0..samples {
            let s = dyn_index.sample(&mut rng).unwrap();
        }
        let runtime = start.elapsed();
        println!("WeightedIndex SingleIncrease [n: {}] Time elapsed: {}ms", t, runtime.as_millis());

        for i in 0..substeps {
            let u = 0;
            weights[u] += 1.0;
            dyn_index.set_weight(u, weights[u]);
        }
        t += substeps;
    }
}

fn main() {
    let n = 10000000;
    let g = 100;
    let samples = 1000000;
    let repeats = 5;

    for r in 0..repeats {
        benchmark_random_increase(n, g, samples);
        benchmark_polya_urn(n, g, samples);
        benchmark_single_increase(n, g, samples);
    }
}