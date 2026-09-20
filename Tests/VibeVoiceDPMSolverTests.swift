import Foundation
@preconcurrency import MLX
import Testing

@testable import MLXAudioTTS

/// Reference trajectory from VibeVoice's bundled `DPMSolverMultistepScheduler`
/// (`vibevoice/schedule/dpm_solver.py`) with `beta_schedule="cosine"`,
/// `prediction_type="v_prediction"` and 20 inference steps. The stand-in model output at
/// step `i`, element `j`, is `((i * 4 + j) % 13 - 6) / 10`, and the sample starts at
/// `(j % 9 - 4) / 10`, so the whole recurrence is reproducible on both sides.
private enum VibeVoiceSolverGolden {
    static let timesteps: [Int] = [999, 949, 899, 849, 799, 749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
    static let trajectory: [Float] = [-0.352111, -0.260194, -0.168276, -0.076359, -0.337060, -0.253178, -0.169296, -0.085414, -0.361489, -0.285487, -0.209486, -0.133484, -0.419247, -0.213088, -0.144925, -0.076761, -0.367587, -0.209745, -0.149396, -0.089046, -0.388211, -0.238653, -0.186103, -0.133553, -0.441065, -0.299331, -0.108316, -0.063555, -0.377773, -0.243838, -0.106556, -0.069576, -0.391802, -0.265651, -0.137237, -0.108032, -0.437278, -0.318743, -0.197975, -0.029876, -0.363216, -0.252441, -0.139434, -0.029074, -0.369994, -0.266968, -0.161709, -0.061226, -0.410571, -0.315283, -0.217761, -0.125127, -0.327269, -0.239708, -0.149912, -0.065013, -0.326450, -0.246604, -0.164521, -0.087340, -0.361956, -0.289810, -0.215426, -0.145947, -0.431028, -0.202879, -0.136171, -0.074375, -0.365480, -0.200599, -0.141533, -0.087389, -0.384380, -0.246870, -0.195369, -0.148806, -0.428271, -0.300428, -0.139977, -0.102705]
}

@Suite("VibeVoice DPM-Solver++")
struct VibeVoiceDPMSolverTests {

    /// `linspace` spacing over 1000 training steps, reversed, dropping the last entry.
    @Test func timestepScheduleMatchesReference() {
        let solver = VibeVoiceDPMSolver()
        solver.setTimesteps(20)
        #expect(solver.timesteps == VibeVoiceSolverGolden.timesteps)
    }

    /// Walks the full 20-step trajectory and compares every intermediate sample. This
    /// covers the first-order opening step, the second-order middle, and the first-order
    /// final step forced by `final_sigmas_type: "zero"` (where sigma is 0 and the update
    /// has to survive a division that goes through infinity).
    @Test func trajectoryMatchesReference() {
        let dim = 4
        let steps = 20
        let solver = VibeVoiceDPMSolver()
        solver.setTimesteps(steps)

        var sample = MLXArray((0 ..< dim).map { Float(($0 % 9) - 4) / 10 }, [1, dim])
        var produced: [Float] = []
        for i in 0 ..< steps {
            let values = (0 ..< dim).map { Float((((i * dim + $0) % 13) - 6)) / 10 }
            let modelOutput = MLXArray(values, [1, dim])
            sample = solver.step(modelOutput: modelOutput, sample: sample)
            produced.append(contentsOf: sample.asArray(Float.self))
        }

        #expect(produced.count == VibeVoiceSolverGolden.trajectory.count)
        var worst: Float = 0
        for (a, b) in zip(produced, VibeVoiceSolverGolden.trajectory) {
            worst = Swift.max(worst, Swift.abs(a - b))
        }
        #expect(worst < 1e-4)
    }
}
