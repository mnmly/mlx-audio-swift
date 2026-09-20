import Foundation
import MLX

/// The inference half of diffusers' `DPMSolverMultistepScheduler`, specialised to the
/// settings VibeVoice constructs it with: `dpmsolver++`, solver order 2, `midpoint`,
/// `linspace` spacing, `final_sigmas_type: "zero"`, and a cosine beta schedule.
///
/// Only sampling is ported — no training, no Karras/Lu sigmas, no SDE variants. The
/// schedule is scalar arithmetic, so it is computed in `Double` on the CPU; only the
/// per-step sample update touches `MLXArray`.
public final class VibeVoiceDPMSolver {
    private let predictionType: String
    private let numTrainTimesteps: Int

    /// `sigmas[i]` for each inference step, with a trailing zero (`final_sigmas_type`).
    private var sigmas: [Double] = []
    /// Discrete training timesteps for each inference step, descending.
    public private(set) var timesteps: [Int] = []

    /// `sigmas` over the full training schedule, indexed by discrete timestep.
    private var trainSigmas: [Double] = []

    /// Converted model outputs from the previous steps; index 1 is the most recent.
    private var modelOutputs: [MLXArray?] = [nil, nil]
    private var stepIndex = 0
    private var lowerOrderNums = 0

    public init(
        numTrainTimesteps: Int = 1000,
        betaSchedule: String = "cosine",
        predictionType: String = "v_prediction"
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        self.predictionType = predictionType

        precondition(
            betaSchedule == "cosine" || betaSchedule == "squaredcos_cap_v2",
            "VibeVoice only ships the cosine beta schedule; got \(betaSchedule)")

        // betas_for_alpha_bar with the Glide cosine transform.
        var alphasCumprod = [Double](repeating: 0, count: numTrainTimesteps)
        var running = 1.0
        for i in 0 ..< numTrainTimesteps {
            let t1 = Double(i) / Double(numTrainTimesteps)
            let t2 = Double(i + 1) / Double(numTrainTimesteps)
            let beta = min(1 - alphaBar(t2) / alphaBar(t1), 0.999)
            running *= (1 - beta)
            alphasCumprod[i] = running
        }
        trainSigmas = alphasCumprod.map { ((1 - $0) / $0).squareRoot() }
    }

    private func alphaBar(_ t: Double) -> Double {
        let x = (t + 0.008) / 1.008 * Double.pi / 2
        return cos(x) * cos(x)
    }

    /// Resets the solver and builds the schedule for `steps` inference steps.
    public func setTimesteps(_ steps: Int) {
        // `lambda_min_clipped` is -inf in this configuration, so no timesteps are clipped.
        let last = Double(numTrainTimesteps)
        var ts: [Int] = []
        for i in 0 ... steps {
            let value = (last - 1) * Double(i) / Double(steps)
            ts.append(Int(value.rounded()))
        }
        // `.round()[::-1][:-1]`: reverse, then drop what is now the last element.
        timesteps = ts.reversed().dropLast()

        sigmas = timesteps.map { trainSigmas[$0] }
        sigmas.append(0)  // final_sigmas_type == "zero"

        modelOutputs = [nil, nil]
        stepIndex = 0
        lowerOrderNums = 0
    }

    private func alphaSigma(_ sigma: Double) -> (alpha: Double, sigma: Double) {
        let alpha = 1 / (sigma * sigma + 1).squareRoot()
        return (alpha, sigma * alpha)
    }

    /// Converts the network output to the `x0` prediction that `dpmsolver++` operates on.
    private func convertModelOutput(_ modelOutput: MLXArray, sample: MLXArray) -> MLXArray {
        let (alphaT, sigmaT) = alphaSigma(sigmas[stepIndex])
        switch predictionType {
        case "v_prediction":
            return Float(alphaT) * sample - Float(sigmaT) * modelOutput
        case "epsilon":
            return (sample - Float(sigmaT) * modelOutput) / Float(alphaT)
        case "sample":
            return modelOutput
        default:
            fatalError("Unsupported prediction_type \(predictionType)")
        }
    }

    /// Advances the sample one step. Call once per entry in `timesteps`.
    public func step(modelOutput: MLXArray, sample: MLXArray) -> MLXArray {
        // The last step drops to first order because `final_sigmas_type == "zero"`, and
        // the very first step has no history to build a second-order estimate from. Every
        // step in between is second order: with `solver_order == 2`, diffusers' `elif`
        // chain always selects the second-order branch once those two cases are excluded.
        let lowerOrderFinal = (stepIndex == timesteps.count - 1)

        let converted = convertModelOutput(modelOutput, sample: sample)
        modelOutputs[0] = modelOutputs[1]
        modelOutputs[1] = converted

        // Upcast before the update, as upstream does, to keep the step numerically stable.
        let sampleF32 = sample.asType(.float32)
        let prev: MLXArray
        if lowerOrderNums < 1 || lowerOrderFinal {
            prev = firstOrderUpdate(converted, sample: sampleF32)
        } else {
            prev = secondOrderUpdate(sample: sampleF32)
        }

        if lowerOrderNums < 2 { lowerOrderNums += 1 }
        stepIndex += 1
        return prev.asType(modelOutput.dtype)
    }

    private func firstOrderUpdate(_ modelOutput: MLXArray, sample: MLXArray) -> MLXArray {
        let (alphaT, sigmaT) = alphaSigma(sigmas[stepIndex + 1])
        let (alphaS, sigmaS) = alphaSigma(sigmas[stepIndex])
        let h = (log(alphaT) - log(sigmaT)) - (log(alphaS) - log(sigmaS))
        return Float(sigmaT / sigmaS) * sample
            - Float(alphaT * (exp(-h) - 1)) * modelOutput
    }

    private func secondOrderUpdate(sample: MLXArray) -> MLXArray {
        let (alphaT, sigmaT) = alphaSigma(sigmas[stepIndex + 1])
        let (alphaS0, sigmaS0) = alphaSigma(sigmas[stepIndex])
        let (alphaS1, sigmaS1) = alphaSigma(sigmas[stepIndex - 1])

        let lambdaT = log(alphaT) - log(sigmaT)
        let lambdaS0 = log(alphaS0) - log(sigmaS0)
        let lambdaS1 = log(alphaS1) - log(sigmaS1)

        guard let m0 = modelOutputs[1], let m1 = modelOutputs[0] else {
            fatalError("Second-order update needs two previous model outputs")
        }

        let h = lambdaT - lambdaS0
        let h0 = lambdaS0 - lambdaS1
        let r0 = h0 / h
        let d1 = Float(1 / r0) * (m0 - m1)

        // `midpoint` solver type.
        let common = Float(alphaT * (exp(-h) - 1))
        return Float(sigmaT / sigmaS0) * sample - common * m0 - 0.5 * common * d1
    }
}
