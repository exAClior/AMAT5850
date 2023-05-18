# most of this code is from example of Molly.jl documentation
# https://juliamolsim.github.io/Molly.jl/stable/differentiable/

using Molly, Zygote, Format, LinearAlgebra, GLMakie


# we want to obtian the correct minimal angle for Harmonic Angle potential
# such that the distance between the two angles on the legs are separated
# by a distance of 1.0
const TRUE_DIST = 1.0

sim_steps = 150
atom_mass = 10.0
boundary = CubicBoundary(3.0)
temp = 0.05
n_atoms = 6

start_coords = [
    SVector(0.8, 0.75, 1.5),
    SVector(1.5, 0.70, 1.5),
    SVector(2.3, 0.75, 1.5),
    SVector(0.8, 2.25, 1.5),
    SVector(1.5, 2.20, 1.5),
    SVector(2.3, 2.25, 1.5),
]

# starts stationary
velocities = zero(start_coords)
simulator = VelocityVerlet(dt=0.05, coupling=BerendsenThermostat(temp, 0.5))

function evolve(θ)

    atoms = [Atom(i, 0.0, atom_mass, 0.0, 0.0, false) for i = 1:n_atoms]
    loggers = (coords=CoordinateLogger(Float64, 2),)
    interaction_list = (
        InteractionList2Atoms(
            [1, 2, 4, 5],
            [2, 3, 5, 6],
            [HarmonicBond(100.0, 0.7) for _ = 1:4],
        ), InteractionList3Atoms(
            [1, 4],
            [2, 5],
            [3, 6],
            [
                HarmonicAngle(10.0, θ),
                HarmonicAngle(10.0, θ),
            ],)
    )

    sys = System(
        atoms=atoms,
        coords=deepcopy(start_coords),
        boundary=boundary,
        velocities=deepcopy(velocities),
        specific_inter_lists=interaction_list,
        loggers=loggers,
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    simulate!(sys, simulator, sim_steps)
    visualize(sys.loggers.coords, boundary, "angle.gif")
end

function loss(θ)

    atoms = [Atom(i, 0.0, atom_mass, 0.0, 0.0, false) for i = 1:n_atoms]
    loggers = (coords=CoordinateLogger(Float64, 2),)
    interaction_list = (
        InteractionList2Atoms(
            [1, 2, 4, 5],
            [2, 3, 5, 6],
            [HarmonicBond(100.0, 0.7) for _ = 1:4],
        ), InteractionList3Atoms(
            [1, 4],
            [2, 5],
            [3, 6],
            [
                HarmonicAngle(10.0, θ),
                HarmonicAngle(10.0, θ),
            ],)
    )

    sys = System(
        atoms=atoms,
        coords=deepcopy(start_coords),
        boundary=boundary,
        velocities=deepcopy(velocities),
        specific_inter_lists=interaction_list,
        loggers=loggers,
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    simulate!(sys, simulator, sim_steps)

    mole1_atom_dist = norm(vector(sys.coords[1], sys.coords[3], boundary))
    mole2_atom_dist = norm(vector(sys.coords[4], sys.coords[6], boundary))
    dist_avg = 0.5 * (mole1_atom_dist + mole2_atom_dist)
    loss_val = abs(dist_avg - TRUE_DIST)

    Zygote.ignore() do
        printfmt(
            "θ {:5.1f}°, Final dist avg {:4.2f} | Loss {:5.3f} | ",
            rad2deg(θ), dist_avg, loss_val
        )
    end
    return loss_val
end

function train_ad()
    θlearn = deg2rad(80.0)
    epochs = 30
    learning_rate = 0.05
    for epoch_n in 1:epochs
        printfmt("Epoch {:3d} |", epoch_n)
        grad = gradient(loss, θlearn)[1]
        printfmt("Grad {:6.4}f\n", round(grad; digits=3))

        θlearn -= grad * learning_rate
    end
end


train_ad()

evolve(deg2rad(91.2))
