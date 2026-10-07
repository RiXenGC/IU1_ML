# -*- coding: utf-8 -*-
"""
visualize_sample.py — первый взгляд на датасет Lyft Motion Prediction.

Запуск из корня репозитория (папка kimi):
    python lab6/src/visualize_sample.py

Что делает:
    1. Открывает lab6/data/raw/sample.zarr и печатает, что внутри.
    2. Сохраняет 3 картинки в lab6/figures/:
       fig1_scene_overview.png  — вся сцена целиком: кто куда ехал за 25 секунд
       fig2_frame_snapshot.png  — один кадр: «фотография» дороги сверху
       fig3_ml_task.png         — СУТЬ ЗАДАЧИ: по прошлому предсказать будущее
"""

import os

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
ZARR_PATH = os.path.join("lab6", "data", "raw", "sample.zarr")
FIG_DIR = os.path.join("lab6", "figures")

FRAME_PERIOD = 0.1          # секунд между кадрами (10 Гц)
SCENE_TO_SHOW = 13          # какую сцену рисовать
FRAME_TO_SHOW = 124         # какой кадр сцены (середина ~25-секундной сцены)

# Классы агентов (из документации Lyft). Нам важны только эти четыре:
LABELS = {1: "unknown", 3: "car", 12: "cyclist", 14: "pedestrian"}
COLORS = {"car": "tab:red", "cyclist": "tab:green",
          "pedestrian": "tab:blue", "unknown": "tab:gray"}


# ---------------------------------------------------------------------------
# ШАГ 1. Открыть датасет и понять, что внутри
# ---------------------------------------------------------------------------
def inspect_dataset():
    root = zarr.open(ZARR_PATH, mode="r")

    scenes = root["scenes"]    # сцены: 25-секундные «ролики» с одной машины Lyft
    frames = root["frames"]    # кадры: моменты времени, 10 штук в секунду
    agents = root["agents"]    # наблюдения: «в кадре K виден объект M в точке (x, y)»

    print("=" * 64)
    print("ЧТО ЛЕЖИТ В sample.zarr")
    print("=" * 64)
    print(f"  scenes : {len(scenes):>9,}  (каждая ~25 секунд)")
    print(f"  frames : {len(frames):>9,}  (10 кадров в секунду)")
    print(f"  agents : {len(agents):>9,}  (одна строка = один объект в одном кадре)")
    print()
    print("  Поля одного кадра (frames):")
    for name in frames.dtype.names:
        print(f"    {name}: {frames.dtype[name]}")
    print()
    print("  Поля одного наблюдения (agents):")
    for name in agents.dtype.names:
        print(f"    {name}: {agents.dtype[name]}")

    # Распределение классов: кто вообще ездит вокруг
    probs = agents[:]["label_probabilities"]
    top_class = np.argmax(probs, axis=1)
    print()
    print("  Классы объектов (по всем наблюдениям):")
    for idx, name in LABELS.items():
        share = np.mean(top_class == idx)
        print(f"    {name:<11} {share:6.1%}")
    return root


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def get_scene(root, scene_idx):
    """Достать кадры одной сцены (и ссылку на массив агентов)."""
    frame_start, frame_end = root["scenes"][scene_idx]["frame_index_interval"]
    return root["frames"][frame_start:frame_end], root["agents"]


def agents_of_frame(frame, agents):
    """Какие объекты видны в данном кадре."""
    a_start, a_end = frame["agent_index_interval"]
    return agents[a_start:a_end]


def agent_label(agent):
    """Класс объекта с наибольшей вероятностью."""
    return LABELS.get(int(np.argmax(agent["label_probabilities"])), "unknown")


def rot(yaw):
    """Матрица поворота на угол yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def to_agent_frame(points, center, yaw):
    """Перевести мировые координаты в систему координат агента:
    агент стоит в (0,0) и смотрит вдоль оси x."""
    return (points - center) @ rot(yaw)


def draw_box(ax, center, extent, yaw, **kwargs):
    """Нарисовать машинку-прямоугольник с чёрточкой направления."""
    length, width = extent[0], extent[1]
    box = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                    [-length / 2, -width / 2], [-length / 2, width / 2],
                    [length / 2, width / 2]])
    box = box @ rot(yaw) + center
    ax.plot(box[:, 0], box[:, 1], **kwargs)
    nose = np.array([[0, 0], [length / 2, 0]]) @ rot(yaw) + center
    ax.plot(nose[:, 0], nose[:, 1], **kwargs)


# ---------------------------------------------------------------------------
# ШАГ 2. Рисунок 1: сцена целиком
# ---------------------------------------------------------------------------
def plot_scene_overview(root):
    import matplotlib.pyplot as plt

    frames, agents = get_scene(root, SCENE_TO_SHOW)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Чёрная линия — путь самой машины Lyft (ego)
    ego = frames["ego_translation"][:, :2]
    ax.plot(ego[:, 0], ego[:, 1], color="black", lw=2, label="машина Lyft (ego)")
    ax.plot(ego[0, 0], ego[0, 1], "*", color="black", ms=20)

    # Точки — все остальные участники движения, сгруппированные по track_id
    tracks = {}
    for i in range(len(frames)):
        for a in agents_of_frame(frames[i], agents):
            tid = int(a["track_id"])
            tracks.setdefault((tid, agent_label(a)), []).append(a["centroid"][:2])

    drawn = set()
    for (tid, label), pts in tracks.items():
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], ".", ms=2, color=COLORS[label], alpha=0.6,
                label=label if label not in drawn else None)
        drawn.add(label)

    ax.set_title(f"Сцена №{SCENE_TO_SHOW}: 25 секунд движения, {len(tracks)} объектов\n"
                 f"(чёрное — машина Lyft, точки — траектории всех остальных)")
    ax.set_xlabel("x, метры")
    ax.set_ylabel("y, метры")
    ax.legend()
    ax.set_aspect("equal")
    path = os.path.join(FIG_DIR, "fig1_scene_overview.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# ШАГ 3. Рисунок 2: один кадр
# ---------------------------------------------------------------------------
def plot_frame_snapshot(root):
    import matplotlib.pyplot as plt

    frames, agents = get_scene(root, SCENE_TO_SHOW)
    frame = frames[FRAME_TO_SHOW]
    ego_xy = frame["ego_translation"][:2]

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_box(ax, ego_xy, np.array([4.7, 2.0, 1.8]), 0.0, color="black", lw=2)
    ax.annotate("ego", ego_xy, fontsize=10, weight="bold")

    for a in agents_of_frame(frame, agents):
        if float(np.max(a["label_probabilities"])) < 0.5:  # отсекаем неуверенные детекции
            continue
        label = agent_label(a)
        draw_box(ax, a["centroid"][:2], a["extent"], float(a["yaw"]),
                 color=COLORS[label], lw=1)

    r = 60  # показываем квадрат 120×120 метров вокруг ego
    ax.set_xlim(ego_xy[0] - r, ego_xy[0] + r)
    ax.set_ylim(ego_xy[1] - r, ego_xy[1] + r)
    ax.set_aspect("equal")
    ax.set_title(f"Один кадр (t = {FRAME_TO_SHOW * FRAME_PERIOD:.1f} с): вид сверху\n"
                 f"прямоугольники = машины/люди, чёрточка = куда смотрят")
    path = os.path.join(FIG_DIR, "fig2_frame_snapshot.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# ШАГ 4. Рисунок 3: суть задачи машинного обучения
# ---------------------------------------------------------------------------
def plot_ml_task(root):
    import matplotlib.pyplot as plt

    frames, agents = get_scene(root, SCENE_TO_SHOW)

    # Собираем траектории всех объектов сцены: track_id -> [(кадр, x, y, yaw, класс), ...]
    tracks = {}
    for i in range(len(frames)):
        for a in agents_of_frame(frames[i], agents):
            tracks.setdefault(int(a["track_id"]), []).append(
                (i, *a["centroid"][:2], float(a["yaw"]), int(np.argmax(a["label_probabilities"]))))

    # Ищем МАШИНУ, которая (а) видна достаточно долго, (б) реально ЕХАЛА
    # (считаем смещение именно за 5-секундное окно будущего)
    mid = len(frames) // 2
    best, best_dist = None, 0.0
    for tid, obs in tracks.items():
        idxs = [o[0] for o in obs]
        if sum(x <= mid for x in idxs) < 5 or sum(x > mid for x in idxs) < 40:
            continue
        fut_pts = np.array([[o[1], o[2]] for o in obs if o[0] > mid][:50])
        if len(fut_pts) < 40:
            continue
        dist = np.linalg.norm(fut_pts[-1] - fut_pts[0])
        if dist > best_dist:
            best, best_dist = (tid, obs), dist

    tid, obs = best
    obs = np.array([(i, x, y, yaw) for i, x, y, yaw, _ in obs])
    now = obs[obs[:, 0] <= mid][-1]          # состояние объекта «сейчас»
    cx, cy, yaw0 = now[1], now[2], now[3]

    history = obs[obs[:, 0] <= mid][-11:, 1:3]   # 1 секунда прошлого
    future = obs[obs[:, 0] > mid][:50, 1:3]      # до 5 секунд будущего

    # Переводим в систему координат объекта: он в (0,0), смотрит вдоль x
    history = to_agent_frame(history, np.array([cx, cy]), yaw0)
    future = to_agent_frame(future, np.array([cx, cy]), yaw0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history[:, 0], history[:, 1], "o-", color="tab:blue", lw=2,
            label=f"ПРОШЛОЕ (вход модели): {len(history) * FRAME_PERIOD:.1f} с")
    ax.plot(future[:, 0], future[:, 1], "o-", ms=4, color="tab:green", lw=2,
            label=f"БУДУЩЕЕ (что предсказываем): {len(future) * FRAME_PERIOD:.1f} с")
    ax.plot([0], [0], marker=">", ms=16, color="black", label="объект «сейчас»")
    ax.set_title(f"Суть задачи: объект track_id={tid} проехал {best_dist:.0f} м.\n"
                 f"Модель видит синее и должна нарисовать зелёное")
    ax.set_xlabel("вперёд, метры")
    ax.set_ylabel("влево, метры")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    path = os.path.join(FIG_DIR, "fig3_ml_task.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    root = inspect_dataset()

    print()
    print("Рисую картинки...")
    for p in (plot_scene_overview(root),
              plot_frame_snapshot(root),
              plot_ml_task(root)):
        print("  ", os.path.abspath(p))
    print("Готово!")


if __name__ == "__main__":
    main()
