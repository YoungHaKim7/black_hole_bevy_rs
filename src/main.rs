use bevy::input::mouse::MouseWheel;
use bevy::math::DVec3;
use bevy::math::primitives::Rectangle;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::texture::ImagePlugin;
use bevy::window::CursorMoved;
use rayon::prelude::*;

use bevy::render::mesh::PrimitiveTopology;

use bevy::reflect::TypePath;
use bevy::render::render_resource::{AsBindGroup, ShaderRef};

#[derive(Resource)]
struct GridSettings {
    is_visible: bool,
    size: u32,
    spacing: f32,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            is_visible: true,
            size: 25,
            spacing: 1e10,
        }
    }
}

#[derive(Component)]
struct Grid;

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct GridMaterial {
    #[uniform(0)]
    color: LinearRgba,
}

impl Material for GridMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/grid_material.wgsl".into()
    }
}


// Physical constants
const SPEED_OF_LIGHT: f64 = 299_792_458.0; // m/s (unused but kept for clarity)
const GRAVITATIONAL_CONSTANT: f64 = 6.67430e-11; // m^3 kg^-1 s^-2

// App configuration
const INITIAL_IMAGE_WIDTH: usize = 320;
const INITIAL_IMAGE_HEIGHT: usize = 180;

#[derive(Resource, Clone, Copy)]
struct BlackHole {
    position: DVec3,
    
    schwarzschild_radius_m: f64,
}

impl BlackHole {
    fn new(position: DVec3) -> Self {
        let r_s = 2.0 * GRAVITATIONAL_CONSTANT * 8.54e36 / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);
        Self {
            position,
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
        .insert_resource(BlackHole::new(DVec3::ZERO))
        .add_plugins((DefaultPlugins.set(ImagePlugin::default_nearest()), MaterialPlugin::<GridMaterial>::default()))
        .insert_resource(GridSettings::default())
        .add_systems(Startup, (setup_texture, setup_grid).chain())
        .add_systems(PostStartup, setup_scene)
        .add_systems(
            Update,
            (orbit_controls, toggle_geodesics, raytrace_to_texture, update_grid, toggle_grid),
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

fn setup_grid(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GridMaterial>>,
    settings: Res<GridSettings>,
) {
    let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD);
    update_mesh_geometry(&mut mesh, &settings, &BlackHole::new(DVec3::ZERO));

    commands.spawn((
        MaterialMeshBundle {
            mesh: meshes.add(mesh),
            material: materials.add(GridMaterial { color: Color::WHITE.into() }),
            ..default()
        },
        Grid,
    ));
}

fn update_grid(
    mut meshes: ResMut<Assets<Mesh>>,
    mut query: Query<(&Handle<Mesh>, &mut Visibility), With<Grid>>,
    settings: Res<GridSettings>,
    bh: Res<BlackHole>,
) {
    if !settings.is_changed() && !bh.is_changed() {
        return;
    }
    for (mesh_handle, mut visibility) in query.iter_mut() {
        if settings.is_visible {
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }

        if let Some(mesh) = meshes.get_mut(mesh_handle) {
            update_mesh_geometry(mesh, &settings, &bh);
        }
    }
}

fn update_mesh_geometry(mesh: &mut Mesh, settings: &GridSettings, bh: &BlackHole) {
    let size = settings.size as usize;
    let spacing = settings.spacing;
    let num_vertices = (size + 1) * (size + 1);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut indices = Vec::with_capacity(size * size * 4);

    for z in 0..=size {
        for x in 0..=size {
            let world_x = (x as f32 - size as f32 / 2.0) * spacing;
            let world_z = (z as f32 - size as f32 / 2.0) * spacing;
            let mut y = 0.0;

            let dx = world_x as f64 - bh.position.x;
            let dz = world_z as f64 - bh.position.z;
            let dist = (dx * dx + dz * dz).sqrt();

            if dist > bh.schwarzschild_radius_m {
                let delta_y = 2.0 * (bh.schwarzschild_radius_m * (dist - bh.schwarzschild_radius_m)).sqrt();
                y += delta_y as f32 - 3e10;
            } else {
                y += 2.0 * bh.schwarzschild_radius_m as f32 - 3e10;
            }

            positions.push([world_x, y, world_z]);
        }
    }

    for z in 0..size {
        for x in 0..size {
            let i = z * (size + 1) + x;
            indices.push(i as u32);
            indices.push((i + 1) as u32);
            indices.push(i as u32);
            indices.push((i + size + 1) as u32);
        }
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
}

fn toggle_grid(mut settings: ResMut<GridSettings>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::KeyH) {
        settings.is_visible = !settings.is_visible;
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
                if bh.intercept(ray.pos) {
                    col = Vec3::new(1.0, 0.0, 0.0);
                    break;
                }
                ray.rk4_step(d_lambda, bh.schwarzschild_radius_m);
                if ray.pos.length() > escape_r {
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
    pos: DVec3,
    vel: DVec3,
    e: f64, // Conserved energy
    
}

impl GeodesicRay {
    fn from_pos_dir(pos: DVec3, dir: DVec3, rs: f64) -> Self {
        let r = pos.length();
        let theta = (pos.z / r).acos();
        let phi = pos.y.atan2(pos.x);

        // Convert direction into spherical basis
        let dr = theta.sin() * phi.cos() * dir.x
            + theta.sin() * phi.sin() * dir.y
            + theta.cos() * dir.z;
        let dtheta = (theta.cos() * phi.cos() * dir.x
            + theta.cos() * phi.sin() * dir.y
            - theta.sin() * dir.z)
            / r;
        let dphi = (-phi.sin() * dir.x + phi.cos() * dir.y) / (r * theta.sin());

        let f = 1.0 - rs / r;
        let dt_dlambda =
            ((dr * dr) / f + r * r * (dtheta * dtheta + theta.sin().powi(2) * dphi * dphi)).sqrt();
        let e = f * dt_dlambda;

        Self {
            pos,
            vel: DVec3::new(dr, dtheta, dphi),
            e,
        }
    }

    // Returns (d/dλ)[r, θ, φ] and (d/dλ)[dr, dθ, dφ]
    fn geodesic_rhs(&self, rs: f64) -> (DVec3, DVec3) {
        let r = self.pos.length();
        let theta = (self.pos.z / r).acos();
        let (dr, dtheta, dphi) = (self.vel.x, self.vel.y, self.vel.z);

        let f = 1.0 - rs / r;
        let dt_dlambda = self.e / f;

        let d_vel = DVec3::new(
            // d²r/dλ²
            -(rs / (2.0 * r * r)) * f * dt_dlambda.powi(2)
                + (rs / (2.0 * r * r * f)) * dr * dr
                + r * (dtheta.powi(2) + theta.sin().powi(2) * dphi.powi(2)),
            // d²θ/dλ²
            -(2.0 / r) * dr * dtheta + theta.sin() * theta.cos() * dphi.powi(2),
            // d²φ/dλ²
            -(2.0 / r) * dr * dphi - 2.0 * theta.cos() / theta.sin() * dtheta * dphi,
        );

        (self.vel, d_vel)
    }

    fn rk4_step(&mut self, d_lambda: f64, rs: f64) {
        let (k1_pos, k1_vel) = self.geodesic_rhs(rs);

        let mut temp_ray = *self;
        temp_ray.pos += k1_pos * (d_lambda / 2.0);
        temp_ray.vel += k1_vel * (d_lambda / 2.0);
        let (k2_pos, k2_vel) = temp_ray.geodesic_rhs(rs);

        let mut temp_ray = *self;
        temp_ray.pos += k2_pos * (d_lambda / 2.0);
        temp_ray.vel += k2_vel * (d_lambda / 2.0);
        let (k3_pos, k3_vel) = temp_ray.geodesic_rhs(rs);

        let mut temp_ray = *self;
        temp_ray.pos += k3_pos * d_lambda;
        temp_ray.vel += k3_vel * d_lambda;
        let (k4_pos, k4_vel) = temp_ray.geodesic_rhs(rs);

        self.pos += (d_lambda / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos);
        self.vel += (d_lambda / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel);

        let r = self.pos.length();
        let theta = (self.pos.z / r).acos();
        let phi = self.pos.y.atan2(self.pos.x);

        self.pos = DVec3::new(
            r * theta.sin() * phi.cos(),
            r * theta.sin() * phi.sin(),
            r * theta.cos(),
        );
    }
}

impl Copy for GeodesicRay {}

impl Clone for GeodesicRay {
    fn clone(&self) -> Self {
        *self
    }
}
