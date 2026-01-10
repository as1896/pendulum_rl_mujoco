import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

###########################################################################################################################
# MujocoEnv
#   mujocoモデルの読み込み, シミュレーション, 描画などの土台のフレームワークを提供
# utils
#   学習器(SB3など)が環境を保存・複製しやすくする補助機能を適用
###########################################################################################################################

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 5.0,
    "azimuth": 90.0,
    "elevation": 0.0,
    "lookat": np.array([0.0, 0.0, 1.0]),
}
"""
    human描画のカメラ初期姿勢
        trackbodyid     :   追従するbody
            0   :   通常worldか最初のbodyに追従
            -1  :   追従しない
        distance        :   注視点からの距離
        azimuth         :   水平方向の角度(deg)
        elevation       :   上下方向の角度(deg)
        lookat          :   注視点（指定可能）

"""

class PendulumEnv(MujocoEnv, utils.EzPickle):
    """
    My inverted pendulum env

    - action        :
    - obs           :   [x, theta, dx, dtheta]
    - reward        :
    - termminate    :   |theta| > theta_threshold or non-finite obs
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]
    }
    """
        render_modes
            human       :   ウィンドウ表示
            rgb_array   :   画像配列で返す
            depth_array :   depth画像
            rgbd_tuple  :   (rgb, depth)のタプルで返す
    """

    def __init__(
            self,
            xml_file: str = "../model/pendulum.xml",
            frame_skip: int = 2,
            default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
            reset_noise_scale: float = 0.01,
            theta_threshold: float = 0.2,
            x_threshold: float | None = None,
            action_scale: float = 50.0,
            **kwargs,
    ):
    
        """
            xml_file                :   MJCFのパス
            frame_skip              :   1回のstepで物理ステップを何回回すか
            default_camera_config   :   human描画時のカメラ初期設定   
            reset_noise_scale       :   reset時に状態を変更するノイズ(毎回違う状態にする)
            theta_threshold         :   棒がこの角度を超えたら失敗(terminated)にする閾値
            x_threshold             :   棒がこの距離を超えたら失敗(terminated)にする閾値
            action_scale            :   学習側のaction([-1,1])を実際の制御入力に変換する係数
            **kwargs                :   MujocoEnv側へ渡したい追加引数を受け取るための要素
        """
        
        # 設定した環境を同じ設定で再生成できるようにコンストラクタ引数を保存する
        # 基本は学習に関係するパラメータを再度同じように入れる
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            reset_noise_scale,
            theta_threshold,
            x_threshold,
            action_scale,
            **kwargs
        )

        # パラメータをselfに保存
        self._reset_noise_scale = float(reset_noise_scale)
        self._theta_threshold = float(theta_threshold)
        self._x_threshold = x_threshold
        self._action_scale = float(action_scale)

        # 観測空間の定義
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        """
            Box()
                low     :   最低値
                high    :   最高値
                shape   :   次元数
                dtype   :   データ型
        """

        # MujocoEnvの初期化(ここでMJCFの読み込み)
        # これを実行後, self.model, self.dataが存在
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        """
            xml_file                :   MJCFのパスを読み込む
                self.model, self.dataを作成
            frame_skip              :   1回のstepで物理ステップを何回回すか
                self.dtを作成
            observation_space       :   観測空間を定義する
            default_camera_config   :   human描画時のカメラ初期設定
        """

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # joint id / dof index をキャッシュ（毎stepで名前検索しない）
        jid_x = self.model.joint("x_slide").id
        jid_th = self.model.joint("pend_hinge").id

        # qpos, qvelのアドレスを指定
        self._iq_x  = self.model.jnt_qposadr[jid_x]
        self._iv_x  = self.model.jnt_dofadr[jid_x]
        self._iq_th = self.model.jnt_qposadr[jid_th]
        self._iv_th = self.model.jnt_dofadr[jid_th]

        # 参考: 観測の構造（ツール/ラッパが使うことがある）
        self.observation_structure = {
            "x": 1,
            "theta": 1,
            "dx": 1,
            "dtheta": 1,
        }

    # シミュレーション空間から必要な変数を返す関数
    def _get_obs(self):
        x      = float(self.data.qpos[self._iq_x])
        theta  = float(self.data.qpos[self._iq_th])
        dx     = float(self.data.qvel[self._iv_x])
        dtheta = float(self.data.qvel[self._iv_th])

        return np.array([x, theta, dx, dtheta], dtype=np.float64)

    def step(self, action):

        # ctrlに渡す値を作成
        # action shape を安定化（SB3等が (1,) を渡す想定）
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        a = np.clip(a[0], -1.0, 1.0)  # 学習側は [-1,1] で統一しても良い

        # ゲインを掛けて値を調整
        ctrl = a * self._action_scale

        # MujocoEnv の do_simulation は self.data.ctrl に書いて回してくれる
        self.do_simulation(np.array([ctrl], dtype=np.float64), self.frame_skip)
        """
            frame_skip回シミュレーションを回す
            シミュレーションを回す前に制御入力を渡す

            self.data.ctrl[:] = ctrl_array
            mj_step()をframe_skip回実行
        """

        # 状態を取得する[x, theta, dx, dtheta]
        obs = self._get_obs()

        # 終了判定
        # フラグ初期化
        terminated = False

        # obs(観測)に1つでもNaN/infが混じったら終了
        if not np.isfinite(obs).all():
            terminated = True

        # thetaが閾値を超えたら終了
        if abs(obs[1]) > self._theta_threshold:   # theta
            terminated = True

        # x が閾値を超えたかつ閾値がNoneでないときに終了
        if self._x_threshold is not None and abs(obs[0]) > float(self._x_threshold):  # x
            terminated = True

        # 報酬：生存 +1（公式と同じ）
        reward = 1.0 if not terminated else 0.0

        info = {
            "reward_survive": reward,
            "ctrl": float(ctrl),
        }

        # render_modeがhumanであるときに描画
        if self.render_mode == "human":
            self.render()

        # truncation は TimeLimit で扱う想定
        return obs, reward, terminated, False, info

    def reset_model(self):
        # 初期状態にノイズを足す（公式と同様）
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # ノイズを加えたあとのqpos
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )

        # ノイズを加えたあとのqvel
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )

        self.set_state(qpos, qvel)
        return self._get_obs()