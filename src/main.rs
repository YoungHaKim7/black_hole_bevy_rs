use bevy::input::mouse::MouseWheel;
use bevy::math::DVec3;
use bevy::math::primitives::Rectangle;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::texture::ImagePlugin;
use bevy::window::CursorMoved;
use rayon::prelude::*;

// Physical constants
const SPEED_OF_LIGHT: f64 = 299_792_458.0; // m/s (unused but kept for clarity)
const GRAVITATIONAL_CONSTANT: f64 = 6.67430e-11; // m^3 kg^-1 s^-2

// App configuration
const INITIAL_IMAGE_WIDTH: usize = 320;
const INITIAL_IMAGE_HEIGHT: usize = 180;

#[derive(Resource, Clone, Copy)]
struct BlackHole {
    position: DVec3,
    mass_kg: f64,
    schwarzschild_radius_m: f64,
}

impl BlackHole {
    fn new(position: DVec3, mass_kg: f64) -> Self {
        let r_s = 2.0 * GRAVITATIONAL_CONSTANT * mass_kg / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);
        Self {
            position,
            mass_kg,
            schwarzschild_radius_m: r_s,
        }
    }
    fn intercept(&self, p: DVec3) -> bool {
        p.distance_squared(self.position)
            < self.schwarzschild_radius_m * self.schwarzschild_radius_m
    }
}

#[derive(Resource)]
struct CameraOrbit {
    target: DVec3,
    radius_m: f64,
    azimuth: f64,
    elevation: f64,
    dragging: bool,
    last_cursor: Vec2,
}

impl Default for CameraOrbit {
    fn default() -> Self {
        Self {
            target: DVec3::ZERO,
            radius_m: 6.34194e10,
            azimuth: 0.0,
            elevation: std::f64::consts::PI / 2.0,
            dragging: false,
            last_cursor: Vec2::ZERO,
        }
    }
}

impl CameraOrbit {
    fn position(&self) -> DVec3 {
        let el = self.elevation.clamp(0.01, std::f64::consts::PI - 0.01);
        DVec3::new(
            self.radius_m * el.sin() * self.azimuth.cos(),
            self.radius_m * el.cos(),
            self.radius_m * el.sin() * self.azimuth.sin(),
        )
    }
}

#[derive(Resource)]
struct RenderTargetSize(pub UVec2);

#[derive(Resource)]
struct RayTexture(Handle<Image>);

#[derive(Resource)]
struct UseGeodesics(pub bool);

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(CameraOrbit::default())
        .insert_resource(UseGeodesics(false))
        .insert_resource(RenderTargetSize(UVec2::new(
            INITIAL_IMAGE_WIDTH as u32,
            INITIAL_IMAGE_HEIGHT as u32,
        )))
        .insert_resource(BlackHole::new(DVec3::ZERO, 8.54e36))
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_systems(Startup, setup_texture)
        .add_systems(PostStartup, setup_scene)
        .add_systems(
            Update,
            (orbit_controls, toggle_geodesics, raytrace_to_texture),
        )
        .run();
}

fn setup_texture(
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
    size: Res<RenderTargetSize>,
) {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.0.x,
            height: size.0.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::RENDER_ATTACHMENT;
    let handle = images.add(image);
    commands.insert_resource(RayTexture(handle.clone()));
}

fn setup_scene(
    mut commands: Commands,
    textures: Res<RayTexture>,
    size: Res<RenderTargetSize>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    orbit: Res<CameraOrbit>,
) {
    // Show the ray-traced image on a quad in front of the camera
    // Create a plane and apply the texture via Pbr material
    let width = size.0.x as f32;
    let height = size.0.y as f32;
    let aspect = width / height;

    let plane_size = Vec2::new(2.0 * aspect, 2.0);

    // Create a simple unlit material that uses the texture
    let mut mat = StandardMaterial::from_color(Color::WHITE);
    mat.base_color_texture = Some(textures.0.clone());
    mat.unlit = true;
    let mat_handle = materials.add(mat);

    let mut mesh = Mesh::from(Rectangle::new(plane_size.x, plane_size.y));
    // Ensure UVs exist
    if mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_none() {
        let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }
    let mesh_handle = meshes.add(mesh);

    commands.spawn(PbrBundle {
        mesh: mesh_handle,
        material: mat_handle,
        transform: Transform::from_xyz(0.0, 0.0, -1.0),
        ..default()
    });

    // Place camera at orbit position looking at target
    let cam_pos = orbit.position().as_vec3();
    commands.spawn(Camera3dBundle {
        transform: Transform::from_translation(cam_pos).looking_at(orbit.target.as_vec3(), Vec3::Y),
        ..default()
    });
}

fn orbit_controls(
    mut orbit: ResMut<CameraOrbit>,
    mut query: Query<&mut Transform, With<Camera>>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    mut cursor_moved: EventReader<CursorMoved>,
    mut mouse_wheel: EventReader<MouseWheel>,
) {
    let orbit_speed = 0.01_f32;
    let zoom_speed = 25e9_f64;

    // Mouse button state
    if mouse_button_input.just_pressed(MouseButton::Left) {
        orbit.dragging = true;
    }
    if mouse_button_input.just_released(MouseButton::Left) {
        orbit.dragging = false;
    }

    // Mouse move
    for ev in cursor_moved.read() {
        let pos = ev.position;
        if orbit.dragging {
            let dx = pos.x - orbit.last_cursor.x;
            let dy = pos.y - orbit.last_cursor.y;
            orbit.azimuth += (dx * orbit_speed) as f64;
            orbit.elevation -= (dy * orbit_speed) as f64;
            orbit.elevation = orbit.elevation.clamp(0.01, std::f64::consts::PI - 0.01);
        }
        orbit.last_cursor = pos;
    }

    // Mouse wheel zoom
    for ev in mouse_wheel.read() {
        orbit.radius_m -= (ev.y as f64) * zoom_speed;
        orbit.radius_m = orbit.radius_m.clamp(1e10, 1e12);
    }

    // Update camera transform
    let cam_pos = orbit.position().as_vec3();
    if let Ok(mut tf) = query.get_single_mut() {
        *tf = Transform::from_translation(cam_pos).looking_at(orbit.target.as_vec3(), Vec3::Y);
    }
}

fn toggle_geodesics(mut use_geo: ResMut<UseGeodesics>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::KeyG) {
        use_geo.0 = !use_geo.0;
        info!("Geodesics: {}", if use_geo.0 { "ON" } else { "OFF" });
    }
}

fn raytrace_to_texture(
    textures: Res<RayTexture>,
    mut images: ResMut<Assets<Image>>,
    size: Res<RenderTargetSize>,
    orbit: Res<CameraOrbit>,
    use_geodesics: Res<UseGeodesics>,
    bh: Res<BlackHole>,
) {
    let Some(img) = images.get_mut(&textures.0) else {
        return;
    };
    let width = size.0.x as usize;
    let height = size.0.y as usize;

    // Camera basis
    let cam_pos = orbit.position();
    let forward = (orbit.target - cam_pos).normalize();
    let right = forward.cross(DVec3::Y).normalize();
    let up = right.cross(forward);
    let fov_y = 60.0_f64.to_radians();
    let aspect = width as f64 / height as f64;
    let tan_half_fov = (fov_y * 0.5).tan();

    let mut pixels: Vec<u8> = vec![0; width * height * 4];

    pixels.par_chunks_mut(4).enumerate().for_each(|(idx, px)| {
        let x = idx % width;
        let y = idx / width;
        let u = (2.0 * ((x as f64 + 0.5) / width as f64) - 1.0) * aspect * tan_half_fov;
        let v = (1.0 - 2.0 * ((y as f64 + 0.5) / height as f64)) * tan_half_fov;
        let dir = (right * u + up * v + forward).normalize();

        // Ray march / geodesic integrate
        let color = if !use_geodesics.0 {
            // simple sphere test (Euclidean)
            let oc = cam_pos - bh.position;
            let b = 2.0 * oc.dot(dir);
            let c0 = oc.length_squared() - bh.schwarzschild_radius_m * bh.schwarzschild_radius_m;
            let disc = b * b - 4.0 * c0;
            if disc > 0.0 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::ZERO
            }
        } else {
            // Full null geodesic, simplified port: 3D spherical coordinates
            let mut ray = GeodesicRay::from_pos_dir(cam_pos, dir, bh.schwarzschild_radius_m);
            let max_steps = 2_000;
            let d_lambda = 1e7_f64;
            let escape_r = 1e14_f64;
            let mut col = Vec3::ZERO;
            for _ in 0..max_steps {
                if bh.intercept(DVec3::new(ray.x, ray.y, ray.z)) {
                    col = Vec3::new(1.0, 0.0, 0.0);
                    break;
                }
                ray.rk4_step(d_lambda, bh.schwarzschild_radius_m);
                if ray.r > escape_r {
                    break;
                }
            }
            col
        };

        px[0] = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
        px[1] = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
        px[2] = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
        px[3] = 255;
    });

    // Write pixels
    img.data = pixels;
}

// ---------------- Geodesic integration (Schwarzschild, null) -----------------

struct GeodesicRay {
    x: f64,
    y: f64,
    z: f64,
    r: f64,
    theta: f64,
    phi: f64,
    dr: f64,
    dtheta: f64,
    dphi: f64,
    e: f64,
    l: f64,
}

impl GeodesicRay {
    fn from_pos_dir(pos: DVec3, dir: DVec3, rs: f64) -> Self {
        let x = pos.x;
        let y = pos.y;
        let z = pos.z;
        let r = (x * x + y * y + z * z).sqrt();
        let theta = (z / r).acos();
        let phi = y.atan2(x);

        // Convert direction into spherical basis
        let dx = dir.x;
        let dy = dir.y;
        let dz = dir.z;
        let dr = theta.sin() * phi.cos() * dx + theta.sin() * phi.sin() * dy + theta.cos() * dz;
        let mut dtheta =
            (theta.cos() * phi.cos() * dx + theta.cos() * phi.sin() * dy - theta.sin() * dz) / r;
        let mut dphi = (-phi.sin() * dx + phi.cos() * dy) / (r * theta.sin());

        // Conserved quantities
        let l = r * r * theta.sin() * dphi;
        let f = 1.0 - rs / r;
        let dt_dlambda = ((dr * dr) / f
            + r * r * (dtheta * dtheta + theta.sin() * theta.sin() * dphi * dphi))
            .sqrt();
        let e = f * dt_dlambda;

        Self {
            x,
            y,
            z,
            r,
            theta,
            phi,
            dr,
            dtheta,
            dphi,
            e,
            l,
        }
    }

    fn geodesic_rhs(&self, rs: f64) -> ([f64; 3], [f64; 3]) {
        let r = self.r;
        let theta = self.theta;
        let dr = self.dr;
        let dtheta = self.dtheta;
        let dphi = self.dphi;
        let e = self.e;
        let f = 1.0 - rs / r;
        let dt_dlambda = e / f;
        let d1 = [dr, dtheta, dphi];
        let d2x = -(rs / (2.0 * r * r)) * f * dt_dlambda * dt_dlambda
            + (rs / (2.0 * r * r * f)) * dr * dr
            + r * (dtheta * dtheta + theta.sin() * theta.sin() * dphi * dphi);
        let d2y = -(2.0 / r) * dr * dtheta + theta.sin() * theta.cos() * dphi * dphi;
        let d2z = -(2.0 / r) * dr * dphi - 2.0 * theta.cos() / theta.sin() * dtheta * dphi;
        (d1, [d2x, d2y, d2z])
    }

    fn rk4_step(&mut self, d_lambda: f64, rs: f64) {
        let (k1a, k1b) = self.geodesic_rhs(rs);

        self.r += d_lambda * k1a[0];
        self.theta += d_lambda * k1a[1];
        self.phi += d_lambda * k1a[2];
        self.dr += d_lambda * k1b[0];
        self.dtheta += d_lambda * k1b[1];
        self.dphi += d_lambda * k1b[2];

        // Back to Cartesian
        self.x = self.r * self.theta.sin() * self.phi.cos();
        self.y = self.r * self.theta.sin() * self.phi.sin();
        self.z = self.r * self.theta.cos();
    }
}
