import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score, precision_recall_fscore_support, roc_auc_score


class Ledger:
    def __init__(self, epsilon, delta, releases):
        self.delta = delta
        self.orders = np.array([1.25, 1.5, 2, 3, 4, 8, 16, 32, 64, 128, 256])
        self.rdp = np.zeros(len(self.orders))
        lo, hi = .01, 1e5
        for _ in range(100):
            mid = (lo + hi) / 2
            composed = releases * self.orders / (2 * mid ** 2)
            bound = np.min(composed + np.log(1 / delta) / (self.orders - 1))
            if bound > epsilon:
                lo = mid
            else:
                hi = mid
        self.multiplier = hi
        self.events = []

    def release(self, records, clip, rng, name):
        matrix = records.reshape(len(records), -1)
        norm_value = np.linalg.norm(matrix, axis=1)
        clipped = matrix * np.minimum(1, clip / np.maximum(norm_value, 1e-12))[:, None]
        sigma = 2 * clip * self.multiplier
        result = clipped.sum(0) + rng.normal(0, sigma, matrix.shape[1])
        self.rdp += self.orders / (2 * self.multiplier ** 2)
        self.events.append({"mechanism": name, "noise_sd": sigma, "clip": clip})
        return result.reshape(records.shape[1:]), sigma

    def report(self):
        bounds = self.rdp + np.log(1 / self.delta) / (self.orders - 1)
        return {"epsilon": float(bounds.min()), "delta": self.delta, "adjacency": "replace-one record within a fixed client and public fold", "noise_multiplier": self.multiplier, "events": self.events.copy()}


def normalize(x, bound):
    return np.clip(x, -bound, bound) / bound


def dp_predictor(x, y, k, ledger, rng, steps):
    weights = np.zeros((x.shape[1] + 1, k)); xx = np.column_stack([x, np.ones(len(x))])
    for _ in range(steps):
        residual = softmax(xx @ weights, axis=1) - np.eye(k)[y]
        gradients = xx[:, :, None] * residual[:, None]
        total, _ = ledger.release(gradients, 1., rng, "discovery_predictor")
        weights -= .1 * total / max(len(x), 1)
    return weights


def signatures(x, y, weights, projection):
    xx = np.column_stack([x, np.ones(len(x))]); probabilities = softmax(xx @ weights, axis=1)
    residual = probabilities - np.eye(probabilities.shape[1])[y]
    gradient = np.linalg.norm(xx, axis=1) * np.linalg.norm(residual, axis=1)
    return np.column_stack([x @ projection, residual[np.arange(len(y)), y], probabilities.max(1), gradient / np.maximum(1, gradient)])


def mixture_fit(z, y, classes, contexts, ledger, rng, steps):
    dimensions = z.shape[1]
    centers = rng.normal(0, .2, (classes, contexts, dimensions)); variance = np.ones_like(centers)
    for _ in range(steps):
        distance = ((z[:, None] - centers[y]) ** 2 / variance[y]).sum(-1) + np.log(variance[y]).sum(-1)
        assignment = softmax(-.5 * distance, axis=1)
        record = np.zeros((len(z), classes, contexts, 1 + 2 * dimensions))
        features = np.column_stack([np.ones(len(z)), z, z ** 2])
        record[np.arange(len(z)), y] = assignment[:, :, None] * features[:, None]
        summary, _ = ledger.release(record, 2., rng, "context_em")
        count = summary[..., :1].clip(1)
        centers = np.clip(summary[..., 1:1 + dimensions] / count, -3, 3)
        variance = np.clip(summary[..., 1 + dimensions:] / count - centers ** 2, .1, 9)
    return centers, variance


def context_assignments(x, y, classes, args, ledger, rng):
    if args.variant == "client_only":
        return np.ones((len(x), 1)), np.zeros(len(x), dtype=int), 0.
    n = len(x); folds = np.arange(n) % args.folds
    projection = rng.normal(size=(x.shape[1], args.projection)) / np.sqrt(x.shape[1])
    assignments = []
    reference_centers = None
    for refit in range(2):
        all_q = np.zeros((n, args.contexts))
        for fold in range(args.folds):
            training = folds != fold; heldout = ~training
            weights = dp_predictor(x[training], y[training], classes, ledger, rng, args.predictor_steps)
            z_train = signatures(x[training], y[training], weights, projection)
            centers, variance = mixture_fit(z_train, y[training], classes, args.contexts, ledger, rng, args.em_steps)
            canonical = centers.mean(0)
            if reference_centers is None:
                reference_centers = canonical
            _, permutation = linear_sum_assignment(((reference_centers[:, None] - canonical[None]) ** 2).sum(-1))
            z = signatures(x[heldout], y[heldout], weights, projection)
            distance = ((z[:, None] - centers[y[heldout]]) ** 2 / variance[y[heldout]]).sum(-1) + np.log(variance[y[heldout]]).sum(-1)
            all_q[heldout] = softmax(-.5 * distance, axis=1)[:, permutation]
        assignments.append(all_q)
    q = (assignments[0] + assignments[1]) / 2
    agreement = (assignments[0] * assignments[1]).sum(-1)
    entropy = -(q * np.log(q.clip(1e-12))).sum(-1) / max(np.log(args.contexts), 1e-12)
    pilot = np.column_stack([q, q * (1 - entropy[:, None]) * agreement[:, None]])
    released, _ = ledger.release(pilot, 1., rng, "context_reliability")
    count = released[:args.contexts].clip(0); reliability = released[args.contexts:].clip(0)
    valid = count >= args.minimum_context
    if valid.sum() < 2 or reliability.sum() / max(count.sum(), 1) < args.reliability:
        return np.ones((n, 1)), q.argmax(-1), 0.
    q[:, ~valid] = 0; q = q / q.sum(1, keepdims=True).clip(1e-12)
    empty = q.sum(1) == 0; q[empty] = 1 / args.contexts
    uncertainty = 1 - np.clip(reliability.sum() / max(count.sum(), 1), 0, 1)
    return q, q.argmax(-1), float(uncertainty)


def screen(x, y, classes, args, ledger, rng):
    relevance = np.zeros((len(x), x.shape[1], classes))
    relevance[np.arange(len(x)), :, y] = x
    pilot, _ = ledger.release(relevance, 1., rng, "candidate_screening")
    return np.argsort(-np.linalg.norm(pilot, axis=1))[:min(args.candidates, x.shape[1])]


def client_summary(x, y, classes, candidates, args, ledger, rng):
    q, inferred, assignment_uncertainty = context_assignments(x, y, classes, args, ledger, rng)
    x = x[:, candidates]; p = x.shape[1]
    one_hot = np.eye(classes)[y]
    per_record = np.concatenate([x[:, :, None] * one_hot[:, None], np.broadcast_to(one_hot[:, None], (len(x), p, classes)), x[:, :, None], np.ones((len(x), p, 1)), x[:, :, None] ** 2], -1)
    records = q[:, :, None, None] * per_record[:, None]
    summary, sigma = ledger.release(records, 2., rng, "joint_relevance_mass")
    mass = summary[..., 2 * classes + 1].clip(1)
    ex = summary[..., 2 * classes] / mass
    ex2 = summary[..., 2 * classes + 2].clip(0) / mass
    prevalence = np.clip(summary[..., classes:2 * classes] / mass[..., None], .01, .99)
    covariance = summary[..., :classes] / mass[..., None] - ex[..., None] * prevalence
    correlations = covariance / np.sqrt(np.maximum(ex2 - ex ** 2, .01)[..., None] * prevalence * (1 - prevalence))
    best_class = np.abs(correlations).argmax(-1)
    relevance = np.take_along_axis(correlations, best_class[..., None], -1).squeeze(-1).clip(-1, 1)
    sampling = np.maximum(ex2 - ex ** 2, .01) / mass
    assignment = assignment_uncertainty / mass
    privacy = (sigma / mass) ** 2 * (1 + relevance ** 2)
    weights = 1 / (sampling + assignment + privacy + 1e-6)
    weights /= weights.sum(0, keepdims=True).clip(1e-12)
    average = (weights * np.abs(relevance)).sum(0)
    latent = (weights * (np.abs(relevance) - average) ** 2).sum(0)
    uncertainty = (weights ** 2 * (sampling + assignment + privacy)).sum(0)
    support = np.abs((weights * np.sign(relevance)).sum(0))
    pair_summary = np.zeros((p, p)); pair_sigma = 4 * ledger.multiplier
    for start in range(0, len(x), 128):
        part = x[start:start + 128]
        squared_norm = (part ** 2).sum(1)
        factor = np.minimum(1, 2 / np.maximum(squared_norm, 1e-12))
        pair_summary += (part * factor[:, None]).T @ part
    pair_summary += rng.normal(0, pair_sigma, pair_summary.shape)
    ledger.rdp += ledger.orders / (2 * ledger.multiplier ** 2)
    ledger.events.append({"mechanism": "redundancy_sketch", "noise_sd": pair_sigma, "clip": 2.})
    redundancy = pair_summary / max(len(x), 1)
    return {"relevance": average, "latent": latent, "variance": uncertainty, "support": support, "candidate": candidates, "redundancy": redundancy, "redundancy_sd": pair_sigma / max(len(x), 1), "inferred": inferred}


def select(summaries, args):
    r = np.array([s["relevance"] for s in summaries]); variance = np.array([s["variance"] for s in summaries]); support = np.array([s["support"] for s in summaries])
    score = r.mean(0) + args.sign_weight * support.mean(0) - args.client_penalty * r.var(0) - args.latent_penalty * np.mean([s["latent"] for s in summaries], 0)
    if args.variant != "no_uncertainty":
        score -= args.confidence * np.sqrt(variance.sum(0) / len(summaries) ** 2)
    candidate = np.arange(len(score))
    if len(candidate) < args.features:
        raise ValueError("DP candidate union is smaller than requested feature subset")
    rho = np.mean([s["redundancy"] for s in summaries], 0)
    sd = np.sqrt(np.sum([s["redundancy_sd"] ** 2 for s in summaries])) / len(summaries)
    phi = np.maximum(0, 1 - norm.ppf(.95) * sd / (np.abs(rho) + 1e-8))
    if args.variant == "no_noise_aware":
        phi = np.ones_like(phi)
    chosen = []
    for _ in range(args.features):
        utility = score[candidate].copy()
        if chosen and args.variant != "no_redundancy":
            utility -= args.redundancy * (phi[np.ix_(candidate, chosen)] * np.abs(rho[np.ix_(candidate, chosen)])).sum(1)
        utility[np.isin(candidate, chosen)] = -np.inf
        chosen.append(int(candidate[np.argmax(utility)]))
    return chosen, score


def run(args):
    data = np.load(args.data, allow_pickle=False)
    x, y, client, split = data["x"], data["y"].astype(int), data["client"], data["split"]
    x = normalize(x, args.bound)
    training = np.flatnonzero(split == "train"); test = np.flatnonzero(split == "test")
    if set(client[training]) & set(client[test]):
        raise ValueError("Selection/training clients must be disjoint from test clients")
    classes = int(y.max()) + 1
    client_ids = np.unique(client[training]); results, subsets, context_scores = [], [], []
    per_client = 4 if args.variant == "client_only" else 2 * args.folds * (args.predictor_steps + args.em_steps) + 4
    accountants = {identifier: Ledger(args.epsilon, args.delta, per_client * args.repeats) for identifier in client_ids}
    started = time.perf_counter()
    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        summaries, ledgers = [], []
        client_started = time.perf_counter()
        candidates = np.unique(np.concatenate([screen(x[training[client[training] == identifier]], y[training[client[training] == identifier]], classes, args, accountants[identifier], rng) for identifier in client_ids]))
        for identifier in client_ids:
            idx = training[client[training] == identifier]
            ledger = accountants[identifier]
            summary = client_summary(x[idx], y[idx], classes, candidates, args, ledger, rng)
            summaries.append(summary); ledgers.append(ledger.report())
            if "context" in data:
                context_scores.append({"repeat": repeat, "client": str(identifier), "ari": adjusted_rand_score(data["context"][idx], summary["inferred"]), "nmi": normalized_mutual_info_score(data["context"][idx], summary["inferred"])})
        client_seconds = time.perf_counter() - client_started
        server_started = time.perf_counter()
        local_chosen, score = select(summaries, args)
        server_seconds = time.perf_counter() - server_started
        chosen = candidates[local_chosen].tolist(); subsets.append(set(chosen))
        classifier = LogisticRegression(max_iter=1000).fit(x[training][:, chosen], y[training])
        pred = classifier.predict(x[test][:, chosen])
        metrics = {"repeat": repeat, "selected": chosen, "macro_f1": f1_score(y[test], pred, average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(y[test], pred), "worst_client_f1": min(f1_score(y[test][client[test] == c], pred[client[test] == c], average="macro", zero_division=0) for c in np.unique(client[test])), "privacy": ledgers, "communication_bytes": int(sum(sum(s[k].nbytes for k in ["relevance", "latent", "variance", "support", "candidate", "redundancy"]) for s in summaries)), "candidate_indices": candidates.tolist(), "scores": score.tolist()}
        if "anchors" in data:
            target = np.zeros(x.shape[1]); target[data["anchors"].astype(int)] = 1
            selected = np.zeros_like(target); selected[chosen] = 1
            precision, recall, f1, _ = precision_recall_fscore_support(target, selected, average="binary", zero_division=0)
            metrics.update(anchor_precision=precision, anchor_recall=recall, anchor_f1=f1)
        metrics["client_seconds"] = client_seconds; metrics["server_seconds"] = server_seconds
        results.append(metrics)
    stability = [len(a & b) / len(a | b) for i, a in enumerate(subsets) for b in subsets[i + 1:]]
    result = {"runs": results, "jaccard_stability": float(np.mean(stability)) if stability else None, "context_recovery": context_scores, "seconds": time.perf_counter() - started}
    if args.attack:
        attack = np.load(args.attack, allow_pickle=False)
        fit = attack["split"] == "train"; held = attack["split"] == "test"
        model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(attack["x"][fit], attack["member"][fit])
        score = model.predict_proba(attack["x"][held])[:, 1]
        auc = roc_auc_score(attack["member"][held], score); pred = score >= .5
        precision, recall, _, _ = precision_recall_fscore_support(attack["member"][held], pred, average="binary", zero_division=0)
        result["attack"] = {"auc": auc, "advantage": 2 * abs(auc - .5), "precision": precision, "recall": recall}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2))
    plt.bar(results[0]["candidate_indices"], results[0]["scores"]); plt.xlabel("Raw feature"); plt.ylabel("Lower-confidence utility"); plt.savefig(output / "feature_scores.png", dpi=180)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--output", required=True); p.add_argument("--attack")
    p.add_argument("--variant", choices=["full", "client_only", "no_uncertainty", "no_noise_aware", "no_redundancy"], default="full")
    for key, val in [("features", 40), ("contexts", 3), ("folds", 3), ("projection", 4), ("predictor-steps", 5), ("em-steps", 5), ("minimum-context", 20), ("candidates", 80), ("repeats", 5), ("seed", 7)]:
        p.add_argument("--" + key, type=int, default=val)
    for key, val in [("epsilon", 4.), ("delta", 1e-5), ("bound", 5.), ("reliability", .1), ("sign-weight", .1), ("client-penalty", 1.), ("latent-penalty", 1.), ("confidence", 1.64), ("redundancy", .1)]:
        p.add_argument("--" + key, type=float, default=val)
    args = p.parse_args()
    if args.contexts < 2 or args.folds < 2 or args.bound <= 0 or args.epsilon <= 0 or not 0 < args.delta < 1:
        p.error("Require contexts/folds>=2, positive bound/epsilon and delta in (0,1)")
    run(args)


if __name__ == "__main__":
    main()
