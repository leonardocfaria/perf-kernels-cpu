import numpy as np
import pytest

from kernels import ckernels, nb, ref

rng = np.random.default_rng(0)
MATMULS = {"nb_naive": nb.matmul_naive, "nb_ikj": nb.matmul_ikj, "nb_tiled": nb.matmul_tiled,
           "c_naive": ckernels.matmul_naive, "c_ikj": ckernels.matmul_ikj, "c_tiled": ckernels.matmul_tiled}
SOFTMAXES = {"nb": nb.softmax, "c": ckernels.softmax}
ATTENTIONS = {"nb": nb.attention, "c": ckernels.attention}


def r32(*shape):
    return rng.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("name", MATMULS)
@pytest.mark.parametrize("M,K,N", [(1, 1, 1), (67, 45, 91), (130, 129, 65), (64, 64, 64)])  # odd sizes cross tile edges
def test_matmul(name, M, K, N):
    A, B = r32(M, K), r32(K, N)
    np.testing.assert_allclose(MATMULS[name](A, B), ref.matmul(A, B), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", SOFTMAXES)
def test_softmax_matches_and_is_stable(name):
    X = r32(33, 257)
    np.testing.assert_allclose(SOFTMAXES[name](X), ref.softmax(X), rtol=1e-5, atol=1e-6)
    big = (r32(4, 100) * 50 + 1000).astype(np.float32)  # naive exp(1000) overflows to inf
    Y = SOFTMAXES[name](big)
    assert np.isfinite(Y).all() and np.allclose(Y.sum(1), 1, atol=1e-5)


@pytest.mark.parametrize("name", ATTENTIONS)
@pytest.mark.parametrize("n,m,d,dv", [(1, 1, 8, 8), (37, 53, 16, 24), (128, 256, 64, 64)])
def test_attention(name, n, m, d, dv):
    Q, K, V = r32(n, d), r32(m, d) * 3, r32(m, dv)  # *3: peaked scores exercise the running-max rescale
    np.testing.assert_allclose(ATTENTIONS[name](Q, K, V), ref.attention(Q, K, V), rtol=1e-4, atol=1e-5)


def test_quick_bench_writes_table_and_roofline(tmp_path):
    import bench

    rows = bench.main(quick=True, out=str(tmp_path))
    assert {r["threads"] for r in rows} == {1, 4}
    assert all(r["gflops"] > 0 for r in rows)
    assert (tmp_path / "roofline.png").stat().st_size > 0 and "| matmul |" in (tmp_path / "bench.md").read_text()
