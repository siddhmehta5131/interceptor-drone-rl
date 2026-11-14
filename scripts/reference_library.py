"""Reference library and attachment rules used by build_full_value_audit.py.

Every URL was returned by a web search on 2026-11-14.  ``url_checked`` records
how strong that check was: a result page that was returned and read is "yes";
an arXiv id taken from the search query only is flagged as such.  Nothing here
claims that a reference confirms a *numeric value*: rules that attach a
reference to an equation or method use the status CITED_METHOD.
"""
from __future__ import annotations

import re

# key -> (citation, URL, url_checked_by_web_search)
LIB = {
    "forster2015": ("Förster, J. (2015). System Identification of the Crazyflie 2.0 Nano Quadrocopter. BSc thesis, ETH Zürich.", "http://hdl.handle.net/20.500.11850/214143", "yes (ETH record)"),
    "nanobench": ("Ullah, S. I., Baca, J. (2026). NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor System Identification, Control, and State Estimation. arXiv:2603.09908; data github.com/syediu/nanobench-iros2026. Confirmed from the paper: Crazyflie 2.1, total flying mass 40.85 g (27 g airframe + markers + charging deck). NOT found in the paper, README, benchmark code or per-run metadata files (checked 2026-11-14): thrust coefficient 2.618e-8 (our own fit) and motor time constant 0.02 s (our assumption) (BUGS #27). The data set logs motor PWM, not rotor speed. The earlier candidate arXiv:2512.14450 (Busetto et al., 45 g Crazyflie 2.1 Brushless, k_F = 3.72e-8) is a different data set and is not the source.", "https://arxiv.org/abs/2603.09908", "yes (2026-11-14: mass confirmed; C_L and motor tau not found)"),
    "faessler2018": ("Faessler, M., Franchi, A., Scaramuzza, D. (2018). Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories. IEEE RA-L 3(2). arXiv:1712.02402. Sec. VII-C, 610 g quadrotor: d_x = 0.544 / 0.491 1/s, d_y = 0.386 / 0.236 1/s (circle / lemniscate); d_z not identified (set to 0).", "https://arxiv.org/abs/1712.02402", "yes (2026-11-14: values read from the paper)"),
    "chadwick1995": ("Chadwick, W. R., Zarchan, P. (1995). Interception of spiraling ballistic missiles. Proc. American Control Conference, vol. 6, p. 4476. doi:10.1109/ACC.1995.532784. Abstract: target weave frequencies of 0.5 to 1 Hz are the range most critical for fast-response proportional navigation interceptors. Abstract only; paper body not read.", "https://doi.org/10.1109/ACC.1995.532784", "yes (abstract only)"),
    "imado1986": ("Imado, F., Miwa, S. (1986). Fighter evasive maneuvers against proportional navigation missile. Journal of Aircraft 23(11):825-830. doi:10.2514/3.45388. Bibliographic record confirmed; content NOT read, so it supports no specific value.", "https://doi.org/10.2514/3.45388", "yes (record only)"),
    "cf21_datasheet": ("Bitcraze AB. Crazyflie 2.1 datasheet (take-off weight 27 g, payload 15 g, 250 mAh 1S LiPo).", "https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf", "yes"),
    "cf_pwm_thrust": ("Bitcraze. PWM to Thrust, crazyflie-firmware documentation.", "https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/pwm-to-thrust/", "yes"),
    "cf_controllers": ("Bitcraze. Controllers (position/velocity/attitude/attitude-rate PID), crazyflie-firmware documentation. NOTE: the 2022.09 version states a 100 Hz loop; the project assumes 500 Hz (S8): verify.", "https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers", "yes"),
    "dryden_w20": ("MathWorks. Dryden Wind Turbulence Model (W20 = wind speed at 20 ft / 6 m).", "https://www.mathworks.com/help/aeroblks/drydenwindturbulencemodelcontinuous.html", "yes"),
    "sola2017": ("Solà, J. (2017). Quaternion kinematics for the error-state Kalman filter. arXiv:1711.02508.", "https://arxiv.org/abs/1711.02508", "yes"),
    "beard2012": ("Beard, R. W., McLain, T. W. (2012). Small Unmanned Aircraft: Theory and Practice. Princeton Univ. Press.", "", "book, no URL"),
    "gravity": ("Standard gravity g0 = 9.80665 m/s^2 (ISA defining constant). Code uses 9.81 (0.03 % higher).", "https://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/atmos/atmos.html", "yes (secondary table)"),
    "zarchan": ("Zarchan, P. Tactical and Strategic Missile Guidance, AIAA Progress in Astronautics and Aeronautics vol. 176 (3rd ed.). N = 3 optimal for unlimited control; a 3-5 range is NOT confirmed from the text.", "", "book, no URL"),
    "barshalom": ("Bar-Shalom, Y., Li, X. R., Kirubarajan, T. (2001). Estimation with Applications to Tracking and Navigation. Wiley; Kalman, R. E. (1960). J. Basic Eng. 82(1).", "", "book, no URL"),
    "ng1999": ("Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. ICML, 278-287.", "https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html", "yes"),
    "pinto2018": ("Pinto, L. et al. (2018). Asymmetric Actor Critic for Image-Based Robot Learning. RSS XIV. arXiv:1710.06542.", "https://arxiv.org/abs/1710.06542", "yes"),
    "johannink2019": ("Johannink, T. et al. (2019). Residual Reinforcement Learning for Robot Control. ICRA. arXiv:1812.03201.", "https://arxiv.org/abs/1812.03201", "yes"),
    "henderson2018": ("Henderson, P. et al. (2018). Deep Reinforcement Learning that Matters. AAAI. arXiv:1709.06560.", "https://arxiv.org/abs/1709.06560", "yes"),
    "agarwal2021": ("Agarwal, R. et al. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS. arXiv:2108.13264.", "https://arxiv.org/abs/2108.13264", "yes"),
    "ppo": ("Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347. App. A Table 3 (p. 10): horizon 2048, Adam 3e-4, 10 epochs, minibatch 64, discount 0.99, GAE 0.95; Table 1 (p. 6): clip 0.2 best; Sec. 6.1: 2 x 64 tanh, no entropy bonus.", "https://arxiv.org/abs/1707.06347", "yes (PDF text read 2026-11-15)"),
    "sac": ("Haarnoja, T. et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor. ICML. arXiv:1801.01290. App. D Table 1: Adam 3e-4, discount 0.99, replay 1e6, 2 x 256 ReLU, minibatch 256, tau 0.005, target update interval 1, gradient steps 1.", "https://arxiv.org/abs/1801.01290", "yes (PDF text read 2026-11-15)"),
    "sac2018b": ("Haarnoja, T. et al. (2018b). Soft Actor-Critic Algorithms and Applications. arXiv:1812.05905. Sec. 5 automatic temperature; App. D Table 1 entropy target -dim(A).", "https://arxiv.org/abs/1812.05905", "yes (PDF text read 2026-11-15)"),
    "td3": ("Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ICML. arXiv:1802.09477. Sec. 6.1: d = 2, target noise N(0, 0.2) clipped (-0.5, 0.5), tau 0.005, exploration N(0, 0.1), 400-300 ReLU, Adam 1e-3, minibatch 100; App. C Table 3: discount 0.99, 1 iteration per time step.", "https://arxiv.org/abs/1802.09477", "yes (PDF text read 2026-11-15)"),
    "sfujim_td3": ("Fujimoto, S. TD3 reference implementation, github.com/sfujim/TD3, commit 34770cc (2023-07-14). TD3.py L18-20 256-256 ReLU, L83/L87 Adam 3e-4, L104 batch 256; main.py L41 start_timesteps 25e3, L44 expl_noise 0.1, L45 batch 256; utils.py L6 buffer 1e6.", "https://github.com/sfujim/TD3/blob/34770ccdcb51df6f6c7d85c1ba7e40d71b940d16/TD3.py", "yes (files read 2026-11-15)"),
    "sb3": ("Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR 22(268):1-8. Defaults read from installed stable_baselines3 2.9.0 (ppo/ppo.py L84-95, sac/sac.py L95-110, td3/td3.py L84-99).", "https://www.jmlr.org/papers/v22/20-1364.html", "yes; installed source read 2026-11-15"),
    "wilson1927": ("Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA 22(158):209-212.", "https://statisticshowto.com/wp-content/uploads/2022/02/wilson_1927.pdf", "yes"),
    "panerati2021": ("Panerati, J. et al. (2021). Learning to Fly - a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control. IROS. arXiv:2103.02142.", "https://arxiv.org/abs/2103.02142", "yes"),
    "isa": ("International Standard Atmosphere (sea level 101.325 kPa, 15 C, 1.225 kg/m^3, a0 = 340.29 m/s); 10 km values (0.4135 kg/m^3, 299.5 m/s) not separately confirmed.", "https://aerodynamics4students.com/properties-of-the-atmosphere/sea-level-conditions.php", "yes (sea level only; secondary)"),
    "cheeseman1955": ("Cheeseman, I. C., Bennett, W. E. (1957; report dated Sept. 1955). The effect of the ground on a helicopter rotor in forward flight. ARC R&M 3021, HMSO. Eq. (2), hover at constant power, image-source method after Betz: T_g/T_inf = 1 / (1 - R^2/(16 Z^2)), R rotor radius, Z hub height.", "https://aerade.cranfield.ac.uk/handle/1826.2/3590", "yes (original scanned PDF read 2026-11-14 and again 2026-11-15: R&M No. 3021, September 1955, eq. (2) on p. 2)"),
    "mcnemar1947": ("McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12(2):153-157. doi:10.1007/BF02295996.", "https://doi.org/10.1007/BF02295996", "yes (DOI resolves to the article, Cambridge Core, 2026-11-14)"),
    "efron1979": ("Efron, B. (1979). Bootstrap methods: another look at the jackknife. Ann. Statist. 7(1):1-26. doi:10.1214/aos/1176344552.", "https://doi.org/10.1214/aos/1176344552", "yes (DOI resolves to the article, Project Euclid, 2026-11-14)"),
}

NA_PREFIX = (
    "interceptor/visualiser", "configs/visualiser",
    "interceptor/utils/logger", "interceptor/utils/model_loader", "interceptor/utils/config_loader",
    "interceptor/common", "run.py", "conftest", "notebooks/",
    "scripts/render", "scripts/plot", "scripts/make_sim_figures", "scripts/make_fixtures",
    "scripts/smoke", "scripts/build_", "scripts/benchmark", "scripts/check_", "scripts/visualise",
    "training/manifest",
    "training/runner", "scripts/trace_liveness",
)

# (location prefix, name regex or None, status, library keys, note)
RULES = [
    ("interceptor/sim/pipeline", None, "CITED_METHOD", ["sola2017", "beard2012", "hairer1993"], "6-DOF / quaternion equations, RK4 integration; this value is an implementation detail"),
    ("interceptor/sim/quaternion", None, "CITED_METHOD", ["sola2017"], "quaternion algebra"),
    ("interceptor/sim/wind", None, "CITED_METHOD", ["dryden", "dryden_w20"], "Dryden form; sigma_w = 0.1*W20 as stated by MathWorks (MIL-F-8785C implementation); original spec not read"),
    ("interceptor/sim/constants", r"^(PH_G_W|g)$", "CITED", ["gravity"], "9.81 used vs 9.80665 standard"),
    ("interceptor/sim/constants", r"^(PH_ARM|PH_M|PH_J|NOM_J|U_BAT_FULL)", "CITED", ["cf21_datasheet", "forster2015"], ""),
    ("interceptor/sim/constants", r"^PH_C_L$", "OWN_FIT", ["nanobench"], "our median fit of m g / (4 Omega^2) over NanoBench hover samples (predecessor repo identify_params.py); NanoBench logs PWM, not rotor speed, so Omega came from our ESC map; the value is not in the NanoBench paper, README or metadata (checked 2026-11-14, BUGS #27)"),
    ("interceptor/sim/constants", r"^PH_K_MOT$", "PROJECT_CHOICE", [], "assumed first-order motor lag; no step-response identification; not in NanoBench (BUGS #27); defended by the motor-lag 0.5x/2x rows of results/tables/quad_limits_repeatability.csv"),
    ("interceptor/sim/constants", r"^(PH_OMEGA_MAX|PH_ETA)$", "PROJECT_CHOICE", [], "labelled 'best guess' in the predecessor simulator (default_demo.py:601, :615); no source found (traced 2026-11-15, BUGS #27); sets the thrust ceiling"),
    ("interceptor/sim/constants", r"^(PH_BAT|PH_C_D|NOM_F|NOM_T)", "CITED_METHOD", ["nanobench", "forster2015", "cf_pwm_thrust"], "motor/ESC/aero identification; fit not independently validated (BUGS #11)"),
    ("interceptor/sim/constants", r"^PH_(KP|KI|KD)$", "CITED_METHOD", ["cf_controllers", "cf_pid_defaults", "cf_rates"], "rate-PID structure from Crazyflie firmware; gains retuned for the simulator units (BUGS F14)"),
    ("configs/airframes", r"gravity", "CITED", ["gravity"], "file says CODATA; 9.80665 is the defined standard gravity, not a CODATA measurement; 9.81 used"),
    ("configs/airframes/aim5", r"air_density|speed_of_sound|speed$", "CITED_METHOD", ["isa"], "ISA 10 km values"),
    ("configs/airframes/aerosonde", r"aspect_ratio", "DERIVED_INTERNAL", [], "AR = b^2/S computed from the sourced span and area"),
    ("configs/quadrotor", r"ground_effect", "PROJECT_CHOICE", ["cheeseman1955"], "linear thrust ramp (EQ-SIX-10), NOT the Cheeseman & Bennett eq. (2) form, which is cited only for the existence and size of the effect; inactive above 0.15 m, so no interception result depends on it"),
    ("interceptor/sim/aero", None, "CITED_METHOD", ["forster2015", "nanobench"], "rotor aerodynamic polynomial; signs hand-flipped (BUGS R-2)"),
    ("interceptor/fc/pid", None, "CITED_METHOD", ["cf_controllers"], "PID structure"),
    ("interceptor/fc/firmware_pid", None, "CITED_METHOD", ["cf_pid_defaults", "cf_rates"], "Crazyflie firmware PID gains and loop rate"),
    ("interceptor/fc/attitude", None, "CITED_METHOD", ["cf_controllers"], "attitude loop structure"),
    ("interceptor/guidance/apn", None, "CITED_METHOD", ["zarchan"], "APN law"),
    ("interceptor/guidance/kinematics", None, "CITED_METHOD", ["zarchan"], "engagement geometry"),
    ("interceptor/guidance/tracker", None, "CITED_METHOD", ["barshalom"], "Kalman tracker"),
    ("interceptor/guidance/seeker", None, "CITED_METHOD", ["barshalom"], "filter form; noise fractions are project choices"),
    ("interceptor/guidance/reward", None, "CITED_METHOD", ["ng1999"], "potential-based shaping form; weights are project choices"),
    ("interceptor/training/asymmetric_policy", None, "CITED_METHOD", ["pinto2018"], "asymmetric actor-critic"),
    ("training/algos", None, "CITED_METHOD", ["sb3", "ppo", "sac", "td3"], "algorithm definitions"),
    ("interceptor/eval/stats", None, "CITED_METHOD", ["wilson1927", "mcnemar1947", "efron1979", "agarwal2021"], "statistics"),
    ("interceptor/guidance/env", r"residual", "CITED_METHOD", ["johannink2019"], "residual-RL form; 0.35 authority is a project choice (P9.3)"),
    ("interceptor/bridge/closed_loop", r"residual", "CITED_METHOD", ["johannink2019"], "residual-RL form"),
    ("interceptor/guidance/platform", None, "DERIVED_INTERNAL", [], "preset duplicate of configs/platforms.yaml, which carries the sources (BUGS #18)"),
    ("interceptor/eval/sweeps", None, "PROJECT_CHOICE", ["agarwal2021"], "sweep grid; reporting follows Agarwal 2021"),
]

PROJECT_PREFIX = (
    "interceptor/", "training/", "scripts/", "configs/",
)


LIB.update({
    "bengio2009": ("Bengio, Y., Louradour, J., Collobert, R., Weston, J. (2009). Curriculum learning. ICML, 41-48. doi:10.1145/1553374.1553380.", "https://mlanthology.org/icml/2009/bengio2009icml-curriculum", "yes"),
    "tobin2017": ("Tobin, J. et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS. arXiv:1703.06907.", "https://arxiv.org/abs/1703.06907", "yes"),
    "peng2018": ("Peng, X. B. et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA. arXiv:1710.06537.", "https://arxiv.org/abs/1710.06537", "yes"),
    "andrychowicz2021": ("Andrychowicz, M. et al. (2021). What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study. ICLR. arXiv:2006.05990. Sec. 3.3 recommendation: always normalise observations; Sec. 3.2 (pp. 5-6): optimal policy width depends on the environment, tanh best and ReLU worst.", "https://arxiv.org/abs/2006.05990", "yes (PDF text read 2026-11-15)"),
    "shaw1985": ("Shaw, R. L. (1985). Fighter Combat: Tactics and Maneuvering. Naval Institute Press. Book confirmed; its treatment of each individual manoeuvre was NOT confirmed.", "https://www.usni.org/press/books/fighter-combat", "yes (book record only)"),
    "gae2016": ("Schulman, J. et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR. arXiv:1506.02438.", "https://arxiv.org/abs/1506.02438", "yes"),
    "nist811": ("NIST SP 811, Guide for the Use of the International System of Units, Appendix B: 1 ft = 0.3048 m exactly.", "https://www.nist.gov/node/1582061", "yes"),
    "goldberg1991": ("Goldberg, D. (1991). What Every Computer Scientist Should Know About Floating-Point Arithmetic. ACM Computing Surveys 23(1). doi:10.1145/103162.103163.", "https://docs.oracle.com/cd/E19957-01/816-2464/ncg_goldberg.html", "yes"),
    "hairer1993": ("Hairer, E., Nørsett, S. P., Wanner, G. (1993). Solving Ordinary Differential Equations I, 2nd ed. Springer, ch. II (classical RK4 of Kutta, Z. Math. Phys. 46:435-453, 1901: order 4, local error O(h^5)). The book was not available to read; the order is measured directly on our simulator instead: tests/unit/test_sim_timestep.py::test_fourth_order_convergence fits a log-log error slope and requires 3.7-4.5 (docs/figures/convergence.png).", "https://doi.org/10.1007/978-3-540-78862-1", "order verified by our own convergence test; book itself not read (library item)"),
    "zipfel2014": ("Zipfel, P. H. (2014). Modeling and Simulation of Aerospace Vehicle Dynamics, 3rd ed. AIAA; CADAC++ AIM5.", "", "book, no URL"),
    "ogata2010": ("Ogata, K. (2010). Modern Control Engineering, 5th ed. Prentice Hall (step-response definitions: 10-90 % rise time, overshoot, settling time).", "", "book, no URL; not checked online"),
})


# Sources read directly on 2026-11-14 (primary files, not search summaries)
LIB.update({
    "far23_337": ("14 CFR 23.337, Limit maneuvering load factors (2011 ed.): positive limit 3.8 normal, 4.4 utility, 6.0 acrobatic; negative 0.4x (normal/utility), 0.5x (acrobatic). US Government Publishing Office.", "https://www.govinfo.gov/content/pkg/CFR-2011-title14-vol1/pdf/CFR-2011-title14-vol1-sec23-337.pdf", "yes (PDF text read)"),
    "cf_rates": ("Bitcraze crazyflie-firmware, stabilizer_types.h: RATE_MAIN_LOOP = 1000 Hz, ATTITUDE_RATE = 500 Hz, POSITION_RATE = 100 Hz.", "https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/interface/stabilizer_types.h", "yes (file read)"),
    "cf_pid_defaults": ("Bitcraze crazyflie-firmware, platform_defaults_cf2.h: rate PID roll/pitch Kp 250, Ki 500, Kd 2.5; yaw Kp 120, Ki 16.7; attitude Kp 6, Ki 3; CF_MASS 0.029 kg; ARM_LENGTH 0.046 m.", "https://github.com/bitcraze/crazyflie-firmware/blob/master/src/platform/interface/platform_defaults_cf2.h", "yes (file read)"),
    "mavsim": ("mavsim_public (Beard & McLain companion code), aerosonde_parameters.py: mass 11.0 kg, gravity 9.81, rho 1.2682, S 0.55, b 2.8956, c 0.18994, Va0 25 m/s.", "https://github.com/byu-magicc/mavsim_public/blob/main/mavsim_python/parameters/aerosonde_parameters.py", "yes (file read)"),
    "dryden": ("Dryden low-altitude model: sigma_w = 0.1 W20, sigma_u/sigma_w = 1/(0.177+0.000823h)^0.4; MIL-F-8785C scale lengths L_w = h, L_u = L_v = h/(0.177+0.000823h)^1.2 (h in ft); MIL-HDBK-1797 uses 2L_w = h. As tabulated by MathWorks.", "https://www.mathworks.com/help/aeroblks/drydenwindturbulencemodelcontinuous.html", "yes, MathWorks page read (it states the block implements MIL-F-8785C); the MIL-F-8785C original and its user guide AFWAL-TR-81-3109 (DTIC ADA119421) could not be downloaded by script (DTIC 403), library/browser item"),
    "usaf_f16": ("US Air Force (Air Combat Command). F-16 Fighting Falcon fact sheet: speed 1,500 mph (Mach 2 at altitude); 'can withstand up to nine G's' with full internal fuel; wingspan 32 ft 8 in (9.8 m); ceiling above 50,000 ft.", "https://www.af.mil/About-Us/Fact-Sheets/Display/Article/104505/f-16-fighting-falcon/", "yes (af.mil blocks scripts; text read 2026-11-14 from Wayback snapshot 20251231004314)"),
    "janes_shahed_irgc": ("Binnie, J. (2023, 17 May). IRGC confirms specs for Shahed-136 attack UAV. Janes. IRGC exhibition poster (Qom): speed 185 km/h, weight 200 kg, warhead 50 kg, range 2,500 km; span 2.5 m / length 3.5 m from a second poster copying a Jan 2022 infographic.", "https://www.janes.com/osint-insights/defence-news/air/irgc-confirms-specs-for-shahed-136-attack-uav", "yes (article text read 2026-11-14)"),
    "echodyne_ita": ("Echodyne Corp. company profile hosted by the US International Trade Administration (Aug 2024): EchoShield angular tracking accuracy <0.5 deg az x <0.5 deg el; EchoGuard azimuth accuracy a little under 1 deg, elevation ~1.5 deg, range ~1-2 m. Statistic (1-sigma or bound) not stated.", "https://www.trade.gov/sites/default/files/2024-08/ECHODYNE.pdf", "yes (PDF text read 2026-11-14)"),
    "cf_controllers": ("Bitcraze. Controllers (position/velocity/attitude/attitude-rate PID), crazyflie-firmware documentation.", "https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers", "yes"),
})

NUM = re.compile(r"(?<![\w.])-?\d+\.?\d*(?:[eE][-+]?\d+)?(?![\w.])")
DIM_CTX = re.compile(r"(zeros|ones|full|eye|reshape|range|round|empty|identity|normal|uniform|"
                     r"linspace|arange|tile|repeat)\(|axis|_DIM\b|\bdim|shape|size=|parents\[", re.I)


def _literals(value: str):
    if "{literals:" in value:
        value = value.split("{literals:", 1)[1].rstrip("} ")
    return NUM.findall(value)


def content_class(name: str, value: str):
    """Classify by what the number is, independent of the file it sits in."""
    lits = _literals(value)
    text = name + " " + value
    if "0.3048" in lits or re.fullmatch(r"FT", name):
        return "UNIT_CONVERSION", ["nist811"], "exact unit definition"
    if lits:
        vals = [abs(float(x)) for x in lits]
        nz = [v for v in vals if v not in (0.0, 1.0, 2.0)]
        guard_ctx = re.search(r"max\(|min\(|[<>]|\+ 1e|norm|eps", text, re.I)
        named_param = re.search(r"DT|dt|K[PID]|gain|lr|rate|coef|sigma|tau", name, re.I)
        if nz and not named_param and (all(v <= 1e-5 for v in nz) or guard_ctx) and all(0 < v <= 1e-3 for v in nz) and all(re.fullmatch(r"-?(1(\.0)?[eE]-\d+|0\.0+1)", x) or float(x) in (0, 1, 2) for x in lits):
            return "NUMERICAL_GUARD", ["goldberg1991"], "division/normalisation guard far below signal scale; not a model parameter"
        ints = all(re.fullmatch(r"-?\d+", x) for x in lits)
        if ints and all(v <= 16 for v in vals) and DIM_CTX.search(text):
            return "MATH_OR_DIMENSION", [], "array dimension, loop count, rounding digits or path depth; structural, not a parameter"
    if re.search(r"\[1\.0, 0\.0, 0\.0, 0\.0\]", value):
        return "MATH_OR_DIMENSION", ["sola2017"], "identity quaternion"
    return None


TUNED = "method cited; the value itself is tuned by this project and must be defended by a sweep/ablation"
# (prefix, name regex, status, keys, note) - checked after RULES
RULES2 = [
    ("configs/training", None, "PROJECT_CHOICE", ["sb3", "ppo", "gae2016", "henderson2018"], "budgets/seeds; " + TUNED),
    ("configs/ablations", None, "PROJECT_CHOICE", ["agarwal2021", "henderson2018"], "ablation grid chosen by the project"),
    ("interceptor/guidance/maneuvers", None, "PROJECT_CHOICE", ["shaw1985", "zarchan", "chadwick1995", "imado1986"], "manoeuvre law constants from notebook v2; weave rates 1.1-2.0 rad/s (0.18-0.32 Hz) are slower than the 0.5-1 Hz critical range of Chadwick & Zarchan (1995); " + TUNED),
    ("interceptor/guidance/targets", None, "PROJECT_CHOICE", ["shaw1985", "chadwick1995"], "target manoeuvre constants from notebook v2; weave rates 1.1-2.0 rad/s (0.18-0.32 Hz) are slower than the 0.5-1 Hz critical range of Chadwick & Zarchan (1995); " + TUNED),
    ("interceptor/sim/constants", r"^(DR_RANGES|WIND_)", "PROJECT_CHOICE", ["tobin2017", "peng2018"], "domain-randomisation ranges; " + TUNED),
    ("interceptor/sim/constants", r"GROUND_EFFECT", "PROJECT_CHOICE", ["cheeseman1955"], "linear thrust ramp (EQ-SIX-10), not the Cheeseman & Bennett eq. (2) form; inactive above 0.15 m, so no interception result depends on it"),
    ("interceptor/guidance/observation", None, "PROJECT_CHOICE", ["andrychowicz2021"], "observation scales set from platform limits (P5.8)"),
    ("interceptor/guidance/airframes", None, "CITED_METHOD", ["beard2012", "zipfel2014"], "point-mass lift/drag and CADAC normal-force formulas"),
    ("interceptor/fc/metrics", None, "CITED_METHOD", ["ogata2010"], "standard step-response metric definitions"),
    ("interceptor/bridge/setpoint", None, "CITED_METHOD", ["cf_controllers"], "derived in docs/mathematics.md section 7; attitude gain from Crazyflie firmware"),
    ("scripts/evaluate", None, "PROJECT_CHOICE", ["agarwal2021", "henderson2018"], "episode counts/seeds for evaluation"),
    ("scripts/sweep", None, "PROJECT_CHOICE", ["agarwal2021"], "sweep grids"),
    ("scripts/residual_analysis", None, "PROJECT_CHOICE", ["agarwal2021"], "analysis settings"),
    ("scripts/collect", None, "PROJECT_CHOICE", ["agarwal2021"], "analysis settings"),
]


def apply_rules(kind: str, loc: str, name: str, status: str, value: str = ""):
    """(status, library keys, note) for a value with no direct source."""
    path = loc.rsplit(":", 1)[0].split("#")[0]
    if kind != "config_sourced":
        cc = content_class(name, value)
        if cc:
            return cc
    for pre, rx, st, keys, note in RULES + RULES2:
        if path.startswith(pre) and (rx is None or re.search(rx, name)):
            return st, keys, note
    if path.startswith(NA_PREFIX) or kind == "code_choice":
        return "NO_CITATION_APPLICABLE", [], "presentation / plumbing / file handling / run bookkeeping; does not affect results"
    if path.startswith(PROJECT_PREFIX):
        return "PROJECT_CHOICE", [], "design or tuning value with no external source; defend by sweep/ablation or state as an assumption"
    return status, [], ""


ANSWER = {
    "CITED": ("LOW", "Taken from the named source."),
    "CITED_PARTIAL": ("MEDIUM", "Source named, but the value is a fit/default/estimate: be ready to show how it was obtained."),
    "CITED_SECONDARY": ("MEDIUM", "Backed only by a secondary web source; replace with a primary datasheet if challenged."),
    "CITED_METHOD": ("LOW", "The equation/method is from the reference; this number follows from it or is an implementation detail."),
    "DERIVED_INTERNAL": ("MEDIUM", "Derived from other project values (docs/mathematics.md or plan decision); only as good as its inputs."),
    "UNIT_CONVERSION": ("LOW", "Exact unit definition."),
    "NUMERICAL_GUARD": ("LOW", "Numerical guard, orders of magnitude below any physical quantity; results insensitive to it."),
    "MATH_OR_DIMENSION": ("LOW", "Mathematical/structural constant (dimension, count, identity); nothing to cite."),
    "NO_CITATION_APPLICABLE": ("LOW", "Presentation or software plumbing; does not affect any reported result."),
    "PROJECT_CHOICE": ("HIGH", "Our own design/tuning choice. No external source exists; say so and point to the sweep/ablation (or run one)."),
    "DATA_DISAGREES": ("HIGH", "Collected real-world data contradicts this value; either change it or state it as a deliberate simplification."),
    "DATA_GAP": ("HIGH", "Searched for real data and found none; state it as an assumption and justify by sensitivity sweep (kill-radius sweep exists: scripts/sweep_robustness.py)."),
    "OWN_FIT": ("MEDIUM", "Fitted by the project to a public data set; the data set is cited, the number is ours. Show the fit script and its inputs; refit planned (P3.4)."),
    "MISSING": ("HIGH", "No source and no justification recorded yet."),
}


# decision id -> (status, library keys, note); data CSVs are the sources for platform decisions
DATA_REFS = "data/threat_platforms.csv, data/interceptor_platforms.csv, data/interceptor_sensors.csv (secondary web sources)"
DECISION_RULES = {
    "D003": ("CITED_METHOD", ["hairer1993"], "RK4 order from the reference; step sizes justified by own convergence plot docs/figures/convergence.png and decision S8"),
    "D014": ("CITED_SECONDARY", [], "hit-to-kill contact rule from sourced spans: Shahed-136 2.5 m (Janes), Orlan-10 3.1 m (Airforce Technology, trade press), F-16C 9.8 m (USAF), Crazyflie 0.092 m (Bitcraze), Aerosonde 2.8956 m (mavsim), AIM5 0.15 m (CADAC); no warhead assumed (data/lethality_and_gload.csv)"),
    "D021": ("CITED", ["far23_337"], "switched 2026-11-14 to +3.8 g / -1.52 g (normal category); also the UAV design requirement in data/lethality_and_gload.csv G01"),
    "D017": ("PROJECT_CHOICE", ["shaw1985"], "manoeuvre names follow air-combat literature; the 16 laws and constants are from notebook v2 (tuned)"),
    "D005": ("CITED_METHOD", ["forster2015", "nanobench", "cf_pwm_thrust"], ""),
    "D007": ("CITED_METHOD", ["nanobench"], "fit not yet done against data (BUGS #11)"),
    "D009": ("CITED_METHOD", ["panerati2021"], "cross-check not yet run (BUGS #12)"),
    "D012": ("CITED_METHOD", ["barshalom"], ""),
    "D013": ("PROJECT_CHOICE", ["echodyne_ita"], "noise model and levels are project choices (earlier source text claiming Vicon/radio measurements was wrong and removed 2026-11-14); nominal angular term 8.727 mrad = EchoShield <0.5 deg bound used as 1-sigma (the statistic is not stated by Echodyne; changed 2026-11-14 from an unsourced 2.5 mrad); P9.1 sweeps noise x0, 0.25, 0.5, 1, 2, 4 (x0.25 is close to the old value, x2 = EchoGuard ~1 deg); base, velocity, acceleration and dropout terms remain project choices"),
    "D015": ("CITED_SECONDARY", [], DATA_REFS),
    "D016": ("CITED_SECONDARY", [], DATA_REFS + "; quad keeps the real Sting/Shahed-136 speed ratio 1.51 (Shahed 185 km/h from the IRGC poster via Janes; CSIS gives 180 km/h; Sting 280 km/h from the manufacturer product page, read by the project author 2026-11-14, page needs a login); fixed-wing target = Orlan-10 cruise 110 km/h = 30.6 m/s (Airforce Technology profile, trade press), speed ratio 40/30.6 = 1.31; quad accel ratio 2.08 is a project choice"),
    "D018": ("CITED_SECONDARY", [], DATA_REFS + "; missile-class 9 g from the USAF F-16 fact sheet (primary)"),
    "D022": ("CITED_SECONDARY", [], DATA_REFS),
    "D023": ("CITED_METHOD", ["ppo", "sac", "td3", "sb3"], "choice pending the 18-run comparison"),
    "D026": ("CITED_METHOD", ["ng1999"], ""),
    "D027": ("PROJECT_CHOICE", [], "reward weights are tuned; justify by sweep"),
    "D028": ("CITED_METHOD", ["pinto2018"], ""),
    "D029": ("CITED_METHOD", ["johannink2019"], "0.35 itself is a project choice, ablated in P9.3"),
    "D031": ("PROJECT_CHOICE", [], "interface chosen by the project (S1); no source"),
    "D032": ("CITED_METHOD", ["panerati2021", "cf_controllers"], ""),
    "D033": ("CITED_METHOD", ["wilson1927", "mcnemar1947", "efron1979", "agarwal2021"], ""),
    "D034": ("CITED_SECONDARY", [], DATA_REFS),
    "D035": ("CITED_SECONDARY", [], DATA_REFS),
}
