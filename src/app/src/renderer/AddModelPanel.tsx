import React, { useState, useEffect } from 'react';
import { useSystem } from './hooks/useSystem';

export interface AddModelInitialValues {
  name: string;
  checkpoint: string;
  recipe: string;
  mmprojOptions?: string[];
  vision?: boolean;
  reranking?: boolean;
  embedding?: boolean;
}

export interface ModelInstallData {
  name: string;
  checkpoint: string;
  recipe: string;
  mmproj?: string;
  reasoning?: boolean;
  vision?: boolean;
  embedding?: boolean;
  reranking?: boolean;
}

interface AddModelPanelProps {
  onClose: () => void;
  onInstall: (data: ModelInstallData) => void;
  initialValues?: AddModelInitialValues;
}

const RECIPE_LABELS: Record<string, string> = {
  'llamacpp': 'Llama.cpp GPU',
  'flm': 'FastFlowLM NPU',
  'oga': 'Ryzen AI LLM',
  // 'oga-hybrid': 'Ryzen AI LLM',
  // 'oga-npu': 'Ryzen AI LLM',
  // 'oga-igpu': 'Ryzen AI LLM',
  // 'whispercpp': 'Whisper.cpp',
  // 'sd-cpp': 'StableDiffusion.cpp',
};

const createEmptyForm = (initial?: AddModelInitialValues) => ({
  name: initial?.name ?? '',
  checkpoint: initial?.checkpoint ?? '',
  recipe: initial?.recipe ?? 'llamacpp',
  mmproj: '',
  reasoning: false,
  vision: initial?.vision ?? false,
  embedding: initial?.embedding ?? false,
  reranking: initial?.reranking ?? false,
});

const AddModelPanel: React.FC<AddModelPanelProps> = ({ onClose, onInstall, initialValues }) => {
  const { supportedRecipes } = useSystem();
  const [form, setForm] = useState(() => createEmptyForm(initialValues));
  const [error, setError] = useState<string | null>(null);

  const mmprojOptions = initialValues?.mmprojOptions ?? [];

  const getMmprojLabel = (filename: string): string =>
    filename.replace(/^mmproj-/i, '').replace(/^model-/i, '').replace(/\.gguf$/i, '');

  useEffect(() => {
    const newForm = createEmptyForm(initialValues);
    if (initialValues?.mmprojOptions && initialValues.mmprojOptions.length > 0) {
      newForm.mmproj = initialValues.mmprojOptions[0];
    }
    setForm(newForm);
    setError(null);
  }, [initialValues]);

  const handleChange = (field: string, value: string | boolean) => {
    setForm(prev => ({ ...prev, [field]: value }));
    setError(null);
  };

  const handleInstall = () => {
    const name = form.name.trim();
    const checkpoint = form.checkpoint.trim();
    const recipe = form.recipe.trim();

    if (!name) {
      setError('Model name is required.');
      return;
    }
    if (!checkpoint) {
      setError('Checkpoint is required.');
      return;
    }
    if (!recipe) {
      setError('Recipe is required.');
      return;
    }
    if (checkpoint.toLowerCase().includes('gguf') && !checkpoint.includes(':')) {
      setError('GGUF checkpoints must include a variant using the CHECKPOINT:VARIANT syntax.');
      return;
    }

    onInstall({
      name,
      checkpoint,
      recipe,
      mmproj: form.mmproj.trim() || undefined,
      reasoning: form.reasoning,
      vision: form.vision,
      embedding: form.embedding,
      reranking: form.reranking,
    });
  };

  const filteredSupportedRecipes = Object.keys(supportedRecipes).filter(r => r in RECIPE_LABELS);
  const recipeOptions = filteredSupportedRecipes.length > 0
    ? filteredSupportedRecipes
    : Object.keys(RECIPE_LABELS);

  const mmprojOptionElements = mmprojOptions.map((f: string) => {
    const label = getMmprojLabel(f);
    return React.createElement('option', { key: f, value: f }, label);
  });

  const showMmproj = mmprojOptions.length > 0 || !initialValues;
  const mmprojField: React.ReactNode = showMmproj
    ? React.createElement(
        'div',
        { className: 'form-subsection' },
        React.createElement(
          'label',
          { className: 'form-label-secondary', title: 'Multimodal projection file for vision models' },
          'mmproj file (Optional)'
        ),
        mmprojOptions.length > 0
          ? React.createElement(
              'select',
              {
                className: 'form-input form-select',
                value: form.mmproj,
                onChange: (e: React.ChangeEvent<HTMLSelectElement>) => handleChange('mmproj', e.target.value),
              },
              ...mmprojOptionElements
            )
          : React.createElement('input', {
              type: 'text',
              className: 'form-input',
              placeholder: 'mmproj-F16.gguf',
              value: form.mmproj,
              onChange: (e: React.ChangeEvent<HTMLInputElement>) => handleChange('mmproj', e.target.value),
            })
      )
    : null;

  return (
    <div className="add-model-form">
      <div className="form-section">
        <label className="form-label" title="A unique name to identify your model in the catalog">
          Model Name
        </label>
        <div className="input-with-prefix">
          <span className="input-prefix">user.</span>
          <input
            type="text"
            className="form-input with-prefix"
            placeholder="Gemma-3-12b-it-GGUF"
            value={form.name}
            onChange={(e) => handleChange('name', e.target.value)}
          />
        </div>
      </div>

      <div className="form-section">
        <label className="form-label" title="Hugging Face model path (repo/model:quantization)">
          Checkpoint
        </label>
        <input
          type="text"
          className="form-input"
          placeholder="unsloth/gemma-3-12b-it-GGUF:Q4_0"
          value={form.checkpoint}
          onChange={(e) => handleChange('checkpoint', e.target.value)}
        />
      </div>

      <div className="form-section">
        <label className="form-label" title="Inference backend to use for this model">Recipe</label>
        <select
          className="form-input form-select"
          value={form.recipe}
          onChange={(e) => handleChange('recipe', e.target.value)}
        >
          <option value="">Select a recipe...</option>
          {recipeOptions.map(recipe => (
            <option key={recipe} value={recipe}>
              {RECIPE_LABELS[recipe] ?? recipe}
            </option>
          ))}
        </select>
      </div>

      <div className="form-section">
        <label className="form-label">More info</label>
        {mmprojField}

        <div className="form-checkboxes">
          <div className="checkbox-item">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={form.reasoning}
                onChange={(e) => handleChange('reasoning', e.target.checked)}
              />
              <span>Reasoning</span>
            </label>
            <p className="checkbox-desc">Model performs multi-step logical thinking. Checkbox is purely for metadata and doesn't affect the output.</p>
          </div>

          <div className="checkbox-item">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={form.embedding}
                onChange={(e) => handleChange('embedding', e.target.checked)}
              />
              <span>Embedding</span>
            </label>
            <p className="checkbox-desc">Outputs numerical vectors that capture semantic meaning. Affects llama.cpp through the <code>--embeddings</code> flag.</p>
          </div>

          <div className="checkbox-item">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={form.reranking}
                onChange={(e) => handleChange('reranking', e.target.checked)}
              />
              <span>Reranking</span>
            </label>
            <p className="checkbox-desc">Reorders a list of inputs based on relevance to a query. Affects llama.cpp through the <code>--reranking</code> flag.</p>
          </div>

          <div className="checkbox-item">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={form.vision}
                onChange={(e) => handleChange('vision', e.target.checked)}
              />
              <span>Vision</span>
            </label>
            <p className="checkbox-desc">Responds to combinations of image and text. Gets auto-selected upon detection of an mmproj file. llama.cpp receives <code>--mmproj &lt;path&gt;</code> for multimodal input.</p>
          </div>
        </div>
      </div>

      {error && <div className="form-error">{error}</div>}

      <div className="form-actions">
        <button className="install-button" onClick={handleInstall}>
          Install
        </button>
        <button className="cancel-button" onClick={onClose}>
          Cancel
        </button>
      </div>
    </div>
  );
};

export default AddModelPanel;
